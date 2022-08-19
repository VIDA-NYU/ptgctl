import inspect
import functools
from typing import Type, TypeVar



class BoundModule:
    '''Acts as a method namespace for a class using a module as the namespace.
    
    The api object will be provided as the first argument to any function accessed from 
    this module.

    .. code-block:: python

        # greetings.py - a module of functions to bind to API

        def hello(api, your_name):
            print(api.name, 'hello', your_name)

        def hi(api):
            print(api.name, 'hi')
        
        # main.py

        class API:
            def __init__(self, name):
                self.name = name

            @property
            def greetings(self):
                import greetings
                return BoundModule(self, greetings)

        API().something.hello('friend')
        API().something.hi()

    '''
    _COPY_OVER = ['__doc__', '__name__', '__file__', '__package__']
    def __init__(self, parent, wrapped):
        self._self = parent
        self._wrapped = wrapped
        keys = getattr(wrapped, '__bind__', None) or getattr(wrapped, '__all__', None) or dir(wrapped)
        candidates = {k: getattr(wrapped, k, None) for k in keys}
        self._public = {
            name: f for name, f in candidates.items() 
            if self._is_public(f)
        }
        self._dir = list(self._public)
        # mimic - idk
        for k in self._COPY_OVER:
            setattr(self, k, getattr(self._wrapped, k))

    def _is_public(self, func):
        return is_public_func(func) and belongs_to_module(self._wrapped, func)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__name__}, {self._dir})'        

    def __dir__(self):
        return self._dir

    def __getattr__(self, name):
        if name not in self._public:
            raise AttributeError(name)
        return self._public[name].__get__(self._self)



def bound_module(get_module):
    '''Helper for bound modules. This will cache per-class for future use.
    
    .. code-block:: python

        class A:
            @bound_module
            def greetings(self):
                import greetings
                return greetings
    '''
    name = f'__{get_module.__name__}'
    @property
    @functools.wraps(get_module)
    def inner(self):
        # only create the bound module once
        try:
            return getattr(self, name)
        except AttributeError:
            bm = BoundModule(self, get_module(self))
            setattr(self, name, bm)
            return bm
    return inner


def is_public_func(func):
    name = getattr(func, '__name__', None)
    return callable(func) and name and not name.startswith('_')

def belongs_to_module(mod, func):
    return inspect.getmodule(func) == mod


# cli namespacing

T = TypeVar("T")

class _NestedMetaClass(type):
    def __init__(self, name, bases, attrs):
        super().__init__(name, bases, attrs)
        self.__attr_name__ = '__{}'.format(name)

    # Works as a property that instantiates a nested class on first access.
    # the created instance will be used for subsequent access.
    def __get__(self: Type[T], instance, owner=None) -> T:
        if instance is None:  # for class based access
            return self  # type: ignore
        try:
            return getattr(instance, self.__attr_name__)
        except AttributeError:
            x = self(instance)
            setattr(instance, self.__attr_name__, x)
            return x

    # don't allow setting this property
    def __set__(self, instance, value):  # This is needed for Fire so that `ismethoddescriptor` is False
        raise TypeError()


class Nest(metaclass=_NestedMetaClass):
    '''This class lets you create method namespaces inside a class while still being able to access the original object.
    
    To access the parent object, use ``self.__``.

    .. code-block:: python

        class MyClass:
            x = 10
            class namespace(Nest):
                y = 15
                def asdf(self):
                    # self.__.x and self.x are the same value btw
                    return self.__.x + self.x + self.y  # 10 + 10 + 15

        import fire
        fire.Fire(MyClass)

    .. code-block:: bash

        python myclass.py nested asdf
    '''

    ROOT_ATTRS = True  # whether to check the parent object for attributes.
    def __init__(self, instance):
        # give it a name, like a function
        self.__name__ = self.__class__.__name__
        # this is the parent instance
        self.__ = instance
        # store the root instance
        root = instance
        if isinstance(instance, Nest):
            root = getattr(instance, '_root_', instance)
        self._root_ = root

    def __getattr__(self, key):
        if not self.ROOT_ATTRS:
            raise KeyError(key)
        return getattr(self.__, key)
