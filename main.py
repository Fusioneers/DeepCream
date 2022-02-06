from DeepCream.deepcream import DeepCream

try:
    dc = DeepCream("data/input", "data/output")
    response_status = dc.start()
    print(response_status)
except BaseException as err:
    print(err)
