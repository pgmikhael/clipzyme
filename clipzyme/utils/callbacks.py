from pytorch_lightning.callbacks import Callback
from clipzyme.utils.registry import get_object


def set_callbacks(trainer, args):
    """
    Set callbacks for trainer, taking into consideration callbacks already set by trainer args.
    Callbacks that are preset by args and perform the same function as those manually selected
    are removed by comparing parent classes between callbacks.

    Parameters
    ----------
    trainer : pl.Trainer
        lightning trainer
    args : Namespace
        global args

    Returns
    -------
    callbacks: list
        complete list of callbacks to be used by trainer
    """
    callbacks = []
    for cback in args.callback_names:
        callbacks.append(get_object(cback, "callback")(args))

    # remove callbacks that are set manually
    redundant_callbacks = []
    for cback in trainer.callbacks:
        parent_cls_preset = get_callback_parent_class(cback)
        for new_cback in callbacks:
            parent_cls_postset = get_callback_parent_class(new_cback)
            if parent_cls_preset == parent_cls_postset:
                redundant_callbacks.append(cback)

    for cback in trainer.callbacks:
        if cback not in redundant_callbacks:
            callbacks.append(cback)

    return callbacks


def get_callback_parent_class(obj):
    """
    Parameters
    ----------
    obj : Callback
        instance of a callback class

    Returns
    -------
    class
        parent class of callback that is the first child of the Callback class
    """
    parent_id = [cls == Callback for cls in obj.__class__.__mro__].index(True)
    parent_cls = obj.__class__.__mro__[parent_id - 1]
    return parent_cls
