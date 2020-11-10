import torch
import time
import os


class LoadModelTimeoutError(TimeoutError):
    pass


class ModelInferTimeoutError(TimeoutError):
    pass


# load the model which was saved by `torch.jit.save()`
def load_model(model_file_path, max_model_load_time, device):
    load_model_start = time.time()
    model = torch.jit.load(model_file_path, map_location='cpu')
    load_model_time = time.time() - load_model_start
    if load_model_time > max_model_load_time:
        raise LoadModelTimeoutError('the model load time is out of limited max time: {}s!'.format(max_model_load_time))
    
    return model, load_model_time


# test in CPU
def get_model_perf_metrics_in_test_loader(model, input_shape, test_sample_num, max_inference_time, device='cpu'):
    model = model.to(device)
    model.eval()
    data = torch.rand(input_shape).unsqueeze(dim=0)
    data = data.to(device)

    start_time = time.time()
    with torch.no_grad():
        for i in range(test_sample_num):            
            if time.time() - start_time > max_inference_time:
                raise ModelInferTimeoutError('the inference time is out of limited max time: {}s!'.format(max_inference_time))
            
            output = model(data)
    
    inference_time = time.time() - start_time
    per_inference_time = inference_time / test_sample_num

    return inference_time, per_inference_time


def save_metrics_to_file(metrics_file_path, is_success, message, load_model_time, inference_time, per_sample_inference_time):
    torch.save({
        'is_success': is_success,
        'message': message,
        'model_load_time': load_model_time,
        'total_inference_time': inference_time,
        'per_sample_inference_time': per_sample_inference_time
    }, metrics_file_path)


def eval_model(model_file_path, metrics_file_path, input_shape, test_sample_num, max_model_load_time, max_inference_time, device='cpu'):
    print('----------------------')
    try:
        is_success = True
        message = 'ok'

        # test that whether the model can load into limited memory by the way
        print('loading model...')
        model, load_model_time = load_model(model_file_path, max_model_load_time, device)

        # if can, measure the performance metrics of the model
        # used memory / inference time in test set
        # no need to measure the test accuracy, it will be measured in the server
        print('using model to do inference...')

        inference_time, per_inference_time = get_model_perf_metrics_in_test_loader(model, input_shape, \
            test_sample_num, max_inference_time, device)

        print('model metrics: ')
        print('model load time: {:.3f}s'.format(load_model_time))
        print('inference time of {} samples: {:.3f}s\n'
              'inference time of per sample: {:.3f}ms'.format(test_sample_num, inference_time, per_inference_time * 1000))

    except LoadModelTimeoutError as e:
        is_success = False
        message = str(e)
        load_model_time, inference_time, per_inference_time = -1, -1, -1

        print(e)
        print('no model loading and inference metrics measured.')

    except ModelInferTimeoutError as e:
        is_success = False
        message = str(e)
        inference_time, per_inference_time = -1, -1

        print(e)
        print('no inference metrics measured.')

    except Exception as e:
        is_success = False
        message = str(e)
        load_model_time, inference_time, per_inference_time = -1, -1, -1

        print(e)
        print('no model loading and inference metrics measured.')
    
    print('----------------------')

    save_metrics_to_file(metrics_file_path, is_success, message, load_model_time, inference_time, per_inference_time)
