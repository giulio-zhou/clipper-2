#include <iostream>
#include <string>
#include <vector>

#include <clipper/datatypes.hpp>
#include <server_http.hpp>

using clipper::VersionedModelId;
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

enum InputType { integer_vec, double_vec, byte_vec, float_vec };
enum OutputType { double_val, int_val };

class AdminServer {
 public:
  AdminServer(int portno, int num_threads) : server(portno, num_threads) {
    AdminServer("0.0.0.0", portno, num_threads);
  }

  AdminServer(std::string address, int portno, int num_threads)
      : server(address, portno, num_threads) {
    /* Add function for admin */
    auto admin_fn = [this](std::shared_ptr<HttpServer::Response> response,
                           std::shared_ptr<HttpServer::Request> request) {
      this->application_admin_endpoint(response, request);
    };
    add_endpoint("^/admin$", "POST", admin_fn);
  }

  /* Assume an input json of the following form:
   * {"name": <new endpoint name>, "models": <array of {"<model_name>":
   * <model_version>} pairs>,
   *  "input_type": <input type string>, "output_type": <output type string>,
   *  "policy": <policy name>, "latency": <latency SLO>}
   */
  void application_admin_endpoint(
      std::shared_ptr<HttpServer::Response> response,
      std::shared_ptr<HttpServer::Request> request);

  /* Dummy function for sending to Redis Model Server */
  void send_to_server(std::string name, std::vector<VersionedModelId> models,
                      InputType input_type, OutputType output_type,
                      std::string policy, long latency);

  void add_endpoint(std::string endpoint, std::string request_method,
                    std::function<void(std::shared_ptr<HttpServer::Response>,
                                       std::shared_ptr<HttpServer::Request>)>
                        endpoint_fn);

  void start_listening();

 private:
  HttpServer server;
};
