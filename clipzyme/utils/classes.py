from abc import ABCMeta
import argparse
from clipzyme.utils.registry import get_object

INITED_OBJ = []


class classproperty(object):
    """
    Method decorator behaves as @classmethod + @property
    """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Nox(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs) -> None:
        super(Nox, self).__init__()

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        pass

    @staticmethod
    def set_args(args) -> None:
        """Set values for class specific args

        Args:
            args (argparse.Namespace): arguments
        """
        pass


def set_nox_type(object_name):
    """
    Build argparse action class for registry items
    Used to add and set object-level args

    Args:
        object_name (str): kind of nox class uses (e.g., dataset, model, lightning)

    Returns:
        argparse.Action: action for specific nox class
    """

    class NoxAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            nargs=None,
            const=None,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=nargs,
                const=const,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )
            self.is_nox_action = True
            self.object_name = object_name

        def __call__(self, parser, namespace, values, option_string=None) -> None:
            setattr(namespace, self.dest, values)

        def add_args(self, parser, values) -> None:
            """
            Add object-level args when an add_argument is called

            Args:
                parser (argparse.parser): nox parser object
                values (Union[list, str]): argument values inputted
            """
            if isinstance(values, list):
                for v in values:
                    obj_val_str = f"{v}_{object_name}"
                    # if object has already been called, conflict arises with add parse called multiple times
                    if obj_val_str not in INITED_OBJ:
                        get_object(v, object_name).add_args(parser)
                        INITED_OBJ.append(obj_val_str)

            elif isinstance(values, str):
                obj_val_str = f"{values}_{object_name}"
                # if object has already been called, conflict arises with add parse called multiple times
                if obj_val_str not in INITED_OBJ:
                    get_object(values, object_name).add_args(parser)
                    INITED_OBJ.append(obj_val_str)

        def set_args(self, args, val) -> None:
            """
            Call object-level set_args method

            Args:
                args (argparse.namespace): global args
                val (Union[list,str]): value for argument
            """
            if isinstance(val, list):
                for v in val:
                    get_object(v, object_name).set_args(args)
            elif isinstance(val, str):
                get_object(val, object_name).set_args(args)

    return NoxAction
