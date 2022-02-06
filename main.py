from DeepCream import deepcream

try:
    response_status = deepcream.start()
    print(response_status)
except BaseException as err:
    print(err)
