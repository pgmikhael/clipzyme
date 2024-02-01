import hashlib

REGISTRIES = {
    "LIGHTNING_REGISTRY": {},
    "DATASET_REGISTRY": {},
    "MODEL_REGISTRY": {},
    "LOSS_REGISTRY": {},
    "METRIC_REGISTRY": {},
    "OPTIMIZER_REGISTRY": {},
    "SCHEDULER_REGISTRY": {},
    "SEARCHER_REGISTRY": {},
    "CALLBACK_REGISTRY": {},
    "INPUT_LOADER_REGISTRY": {},
    "AUGMENTATION_REGISTRY": {},
    "LOGGER_REGISTRY": {},
}


def get_object(object_name, object_type):
    if object_name not in REGISTRIES["{}_REGISTRY".format(object_type.upper())]:
        raise Exception(
            "INVALID {} NAME: {}. AVAILABLE {}".format(
                object_type.upper(),
                object_name,
                REGISTRIES["{}_REGISTRY".format(object_type.upper())].keys(),
            )
        )
    return REGISTRIES["{}_REGISTRY".format(object_type.upper())][object_name]


def register_object(object_name, object_type):
    def decorator(obj):
        REGISTRIES["{}_REGISTRY".format(object_type.upper())][object_name] = obj
        obj.name = object_name
        return obj

    return decorator


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()
