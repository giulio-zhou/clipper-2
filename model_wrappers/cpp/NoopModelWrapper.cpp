#include<vector>
#include "ClipperRPC.h"
#include "NoopModelWrapper.h"

vector<double>* NoopModelWrapper::predict_bytes(
        vector<vector<char> >& input) {
    int i, j;
    double total;
    vector<double> *predictions = new vector<double>(input.size());
    for (i = 0; i < input.size(); i++) {
        total = 0;
        for (j = 0; j < input[i].size(); j++) {
            total += input[i][j];
        }
        (*predictions)[i] = total;
    }
    return predictions;
}

vector<double>* NoopModelWrapper::predict_floats(
        vector<vector<double> >& input) {
    int i, j;
    double total;
    vector<double> *predictions = new vector<double>(input.size());
    for (i = 0; i < input.size(); i++) {
        total = 0;
        for (j = 0; j < input[i].size(); j++) {
            total += input[i][j];
        }
        (*predictions)[i] = total;
    }
    return predictions;
}

vector<double>* NoopModelWrapper::predict_ints(
        vector<vector<uint32_t> >& input) {
    int i, j;
    double total;
    vector<double> *predictions = new vector<double>(input.size());
    for (i = 0; i < input.size(); i++) {
        total = 0;
        for (j = 0; j < input[i].size(); j++) {
            total += input[i][j];
        }
        (*predictions)[i] = total;
    }
    return predictions;
}

int main() {
    std::unique_ptr<Model> model(new NoopModelWrapper());
    ClipperRPC *clipper_rpc_server =
        new ClipperRPC(model, (char *) "127.0.0.1", 6001);
    clipper_rpc_server->serve_forever();
}
