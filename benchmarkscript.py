import os
import subprocess
import glob
import onnx
from onnx import helper, numpy_helper
import mxnet as mx
import time
# import mxnet.contrib.onnx._import as onnx_mxnet

subprocess.call("./setup.sh")

def get_model_input(model_dir):
    # if model_test.model_dir is None:
    #     model_dir = self._prepare_model_data(model_test)
    # else:
    #     model_dir = model_test.model_dir
    # model_pb_path = os.path.join(model_dir, 'model.onnx')
    # model = onnx.load(model_pb_path)
    # model_marker[0] = model
    # prepared_model = self.backend.prepare(model, device)

    # # TODO after converting all npz files to protobuf, we can delete this.
    # for test_data_npz in glob.glob(
    #         os.path.join(model_dir, 'test_data_*.npz')):
    #     test_data = np.load(test_data_npz, encoding='bytes')
    #     inputs = list(test_data['inputs'])
    #     outputs = list(prepared_model.run(inputs))
    #     ref_outputs = test_data['outputs']
    #     self._assert_similar_outputs(ref_outputs, outputs)
    model_inputs = {}
    for test_data_dir in glob.glob(
            os.path.join(model_dir, "test_data_set*")):
        inputs = []
        inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
        for i in range(inputs_num):
            input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))
        # ref_outputs = []
        # ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
        # for i in range(ref_outputs_num):
        #     output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        #     tensor = onnx.TensorProto()
        #     with open(output_file, 'rb') as f:
        #         tensor.ParseFromString(f.read())
        #     ref_outputs.append(numpy_helper.to_array(tensor))
        test_data_name = test_data_dir.split("/")[-1]
        model_inputs.update({test_data_name: inputs})
        # outputs = list(prepared_model.run(inputs))
        # self._assert_similar_outputs(ref_outputs, outputs)
    return model_inputs


def profile_model(model_path, test_data):
    sym, arg_params, aux_params = mx.contrib.onnx.import_model(model_path)
    ctx = mx.cpu()
    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg_params and graph_input not in aux_params]

    log_data = {}
    for test_data_name, inputs in test_data.iteritems():
        data_shapes = []
        for idx, input_name in enumerate(data_names):
                data_shapes.append((input_name, inputs[idx].shape))

        # create a module
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)

        # initializing parameters for calculating result of each individual node
        if arg_params is None and aux_params is None:
            mod.init_params()
        else:
            mod.set_params(arg_params=arg_params, aux_params=aux_params)

        data_forward = []
        for idx, input_name in enumerate(data_names):
            # slice and pad operator tests needs 1 less dimension in forward pass
            # otherwise it will throw an error.
            # for squeeze operator, need to retain shape of input as provided
            val = inputs[idx]
            data_forward.append(mx.nd.array(val))

        start = time.time()
        mod.forward(mx.io.DataBatch(data_forward))
        total_time =  (time.time() - start)*1000
        total_time = "{:.9f}".format(total_time)
        log_data.update({test_data_name: total_time})
    return log_data


if __name__ == '__main__':
    for directory in os.listdir("./models"):
        model_dir = os.path.join("./models", directory)
        if os.path.isdir(model_dir):
            model_path = os.path.join(model_dir, "model.onnx")
            test_data = get_model_input(model_dir)

            profile_data = profile_model(model_path, test_data)

            print(directory)
            for k, v in profile_data.iteritems():
                print('{} {} ms'.format(k, v))

