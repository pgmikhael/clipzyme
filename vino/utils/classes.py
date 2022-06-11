from abc import ABCMeta
import argparse
from vino.utils.registry import get_object


class Vino(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs) -> None:
        super(Vino, self).__init__()

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


def set_vino_type(object_name):
    """
    Build argparse action class for registry items
    Used to add and set object-level args

    Args:
        object_name (str): kind of vino class uses (e.g., dataset, model, lightning)

    Returns:
        argparse.Action: action for specific vino class
    """

    class VinoAction(argparse.Action):
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
            self.is_vino_action = True
            self.object_name = object_name

        def __call__(self, parser, namespace, values, option_string=None) -> None:
            setattr(namespace, self.dest, values)

        def add_args(self, parser, values) -> None:
            """
            Add object-level args when an add_argument is called

            Args:
                parser (argparse.parser): vino parser object
                values (Union[list, str]): argument values inputted
            """
            if isinstance(values, list):
                for v in values:
                    get_object(v, object_name).add_args(parser)
            elif isinstance(values, str):
                get_object(values, object_name).add_args(parser)

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

    return VinoAction
