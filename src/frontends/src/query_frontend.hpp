#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/metrics.hpp>
#include <clipper/query_processor.hpp>
#include <clipper/redis.hpp>

#include <server_http.hpp>
#include "json_util.hpp"

using clipper::Response;
using clipper::FeedbackAck;
using clipper::VersionedModelId;
using clipper::InputType;
using clipper::Input;
using clipper::OutputType;
using clipper::Output;
using clipper::Query;
using clipper::FeedbackQuery;
using clipper_json::json_parse_error;
using clipper_json::json_semantic_error;
using clipper_json::decode_input;
using clipper_json::decode_output;
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

namespace query_frontend {

const std::string GET_METRICS = "^/metrics$";

void respond_http(std::string content, std::string message,
                  std::shared_ptr<HttpServer::Response> response) {
  *response << "HTTP/1.1 " << message
            << "\r\nContent-Length: " << content.length() << "\r\n\r\n"
            << content << "\n";
}

template <class QP>
class RequestHandler {
 public:
  RequestHandler(std::string address, int portno, int num_threads)
      : server_(address, portno, num_threads), query_processor_() {
    clipper::Config& conf = clipper::get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      std::cout << "ERROR: Query frontend connecting to Redis" << std::endl;
      std::cout << "Sleeping 1 second..." << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      std::cout << "ERROR: Query frontend subscriber connecting to Redis"
                << std::endl;
      std::cout << "Sleeping 1 second..." << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    server_.add_endpoint(
        GET_METRICS, "GET",
        [](std::shared_ptr<HttpServer::Response> response,
           std::shared_ptr<HttpServer::Request> /*request*/) {
          clipper::metrics::MetricsRegistry& registry =
              clipper::metrics::MetricsRegistry::get_metrics();
          std::string metrics_report = registry.report_metrics();
          std::cout << "METRICS\n" << metrics_report << std::endl;
          respond_http(metrics_report, "200 OK", response);
        });

    clipper::redis::subscribe_to_application_changes(
        redis_subscriber_,
        [this](const std::string& key, const std::string& event_type) {
          std::cout << "APPLICATION EVENT DETECTED. Key: " << key
                    << ", event_type: " << event_type << std::endl;
          if (event_type == "hset") {
            std::string name = key;
            std::cout << "New application detected: " << key << std::endl;
            auto app_info =
                clipper::redis::get_application_by_key(redis_connection_, key);
            std::vector<VersionedModelId> candidate_models =
                clipper::redis::str_to_models(app_info["candidate_models"]);
            InputType input_type =
                clipper::parse_input_type(app_info["input_type"]);
            OutputType output_type =
                clipper::parse_output_type(app_info["output_type"]);
            std::string policy = app_info["policy"];
            int latency_slo_micros = std::stoi(app_info["latency_slo_micros"]);
            add_application(name, candidate_models, input_type, output_type,
                            policy, latency_slo_micros);
          }
        });
  }

  ~RequestHandler() {
    redis_connection_.disconnect();
    redis_subscriber_.disconnect();
  }

  void add_application(std::string name, std::vector<VersionedModelId> models,
                       InputType input_type, OutputType output_type,
                       std::string policy, long latency_slo_micros) {
    auto predict_fn = [this, name, input_type, policy, latency_slo_micros,
                       models](std::shared_ptr<HttpServer::Response> response,
                               std::shared_ptr<HttpServer::Request> request) {
      try {
        auto prediction =
            decode_and_handle_predict(request->content.string(), name, models,
                                      policy, latency_slo_micros, input_type);
        prediction.then([response](boost::future<Response> f) {
          Response r = f.get();
          std::stringstream ss;
          ss << "qid:" << r.query_id_ << ", predict:" << r.output_.y_hat_;
          std::string content = ss.str();
          respond_http(content, "200 OK", response);
        });
      } catch (const json_parse_error& e) {
        respond_http(e.what(), "400 Bad Request", response);
      } catch (const json_semantic_error& e) {
        respond_http(e.what(), "400 Bad Request", response);
      } catch (const std::invalid_argument& e) {
        respond_http(e.what(), "400 Bad Request", response);
      }
    };
    std::string predict_endpoint = "^/" + name + "/predict$";
    server_.add_endpoint(predict_endpoint, "POST", predict_fn);

    auto update_fn = [this, name, input_type, output_type, policy, models](
        std::shared_ptr<HttpServer::Response> response,
        std::shared_ptr<HttpServer::Request> request) {
      try {
        auto update =
            decode_and_handle_update(request->content.string(), name, models,
                                     policy, input_type, output_type);
        update.then([response](boost::future<FeedbackAck> f) {
          FeedbackAck ack = f.get();
          std::stringstream ss;
          ss << "Feedback received? " << ack;
          std::string content = ss.str();
          respond_http(content, "200 OK", response);
        });
      } catch (const json_parse_error& e) {
        respond_http(e.what(), "400 Bad Request", response);
      } catch (const json_semantic_error& e) {
        respond_http(e.what(), "400 Bad Request", response);
      } catch (const std::invalid_argument& e) {
        respond_http(e.what(), "400 Bad Request", response);
      }
    };
    std::string update_endpoint = "^/" + name + "/update$";
    server_.add_endpoint(update_endpoint, "POST", update_fn);
  }

  /*
   * JSON format for prediction query request:
   * {
   *  "uid" := string,
   *  "input" := [double] | [int] | [string] | [byte] | [float],
   * }
   */
  boost::future<Response> decode_and_handle_predict(
      std::string json_content, std::string name,
      std::vector<VersionedModelId> models, std::string policy,
      long latency_slo_micros, InputType input_type) {
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(json_content.c_str());
    if (!ok) {
      std::stringstream ss;
      ss << "JSON parse error: " << rapidjson::GetParseError_En(ok.Code())
         << " (offset " << ok.Offset() << ")\n";
      throw json_parse_error(ss.str());
    }

    try {
      long uid = d["uid"].GetInt64();
      std::shared_ptr<Input> input = decode_input(input_type, d);
      auto prediction = query_processor_.predict(
          Query{name, uid, input, latency_slo_micros, policy, models});
      return prediction;
    } catch (const rapidjson_exception& e) {
      throw json_semantic_error(e.what());
    }
  }

  /*
   * JSON format for feedback query request:
   * {
   *  "uid" := string,
   *  "input" := [double] | [int] | [string] | [byte] | [float],
   *  "model_name" := string,
   *  "model_version" := int,
   *  "label" := double
   * }
   */
  boost::future<FeedbackAck> decode_and_handle_update(
      std::string json_content, std::string name,
      std::vector<VersionedModelId> models, std::string policy,
      InputType input_type, OutputType output_type) {
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(json_content.c_str());
    if (!ok) {
      std::stringstream ss;
      ss << "JSON parse error: " << rapidjson::GetParseError_En(ok.Code())
         << " (offset " << ok.Offset() << ")\n";
      throw json_parse_error(ss.str());
    }

    try {
      long uid = d["uid"].GetInt64();
      std::shared_ptr<Input> input = decode_input(input_type, d);
      Output output = decode_output(output_type, d);
      auto update = query_processor_.update(FeedbackQuery{
          name, uid, {std::make_pair(input, output)}, policy, models});
      return update;
    } catch (const rapidjson_exception& e) {
      throw json_semantic_error(e.what());
    }
  }

  void start_listening() { server_.start(); }

  /**
   * Returns the number of applications that have been registered
   * with Clipper. This is equivalent to the number of /predict,/update
   * REST endpoint pairs that have been registered with the server.
   * We don't count the /metrics endpoint as it does not serve predictions.
   */
  size_t num_applications() {
    // Subtract one to account for the /metrics endpoint
    size_t count = server_.num_endpoints() - 1;
    assert(count % 2 == 0);
    return count / 2;
  }

 private:
  HttpServer server_;
  QP query_processor_;
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
};

}  // namespace query_frontend
