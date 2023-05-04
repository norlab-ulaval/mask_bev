# Copied from https://github.com/willGuimont/pipeline
# Some Compose and Lambda where taken from:
# https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html
# for a dependency-free pipeline
from typing import Callable


class Lambda:
    """
    Elevate a function into the transform structure
    """

    def __init__(self, f: Callable):
        if not callable(f):
            raise TypeError("Argument lambda should be callable, got {}".format(repr(type(f).__name__)))
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Identity:
    def __call__(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose:
    """
    Compose two transformations
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Tupled:
    """
    Create a tuple by repeating the element
    """

    def __init__(self, n: int):
        self.n = n

    def __call__(self, x):
        return tuple([x for _ in range(self.n)])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MapAll:
    """
    Map a function to all elements of a tuple
    """

    def __init__(self, f: Callable):
        if not callable(f):
            raise TypeError("Argument lambda should be callable, got {}".format(repr(type(f).__name__)))
        self.f = f

    def __call__(self, x):
        return tuple([self.f(e) for e in x])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Tee:
    """
    Duplicate input
    """

    def __call__(self, x):
        return x, x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MapNth:
    """
    Map a function to the nth element
    """

    def __init__(self, f: Callable, n: int):
        if not callable(f):
            raise TypeError("Argument lambda should be callable, got {}".format(repr(type(f).__name__)))
        self.f = f
        self.n = n

    def __call__(self, x):
        before = x[:self.n]
        elem = x[self.n]
        after = x[self.n + 1:]
        return *before, self.f(elem), *after

    def __repr__(self):
        return self.__class__.__name__ + '()'


class First(MapNth):
    """
    Map function to first element of a pair
    """

    def __init__(self, f: Callable):
        super().__init__(f, 0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Second(MapNth):
    """
    Map function to second element of a pair
    """

    def __init__(self, f: Callable):
        super().__init__(f, 1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Third(MapNth):
    """
    Map function to third element of a pair
    """

    def __init__(self, f: Callable):
        super().__init__(f, 2)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Bifunctor:
    """
    Map a function to the first element of a pair and another to the other element
    """

    def __init__(self, f: Callable, g: Callable):
        if not callable(f):
            raise TypeError("Argument lambda should be callable, got {}".format(repr(type(f).__name__)))
        self.f = f
        if not callable(g):
            raise TypeError("Argument lambda should be callable, got {}".format(repr(type(g).__name__)))
        self.g = g

    def __call__(self, x):
        (a, b) = x
        return self.f(a), self.g(b)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Both:
    """
    Map function to both element of a pair
    """

    def __init__(self, f: Callable):
        if not callable(f):
            raise TypeError("Argument lambda should be callable, got {}".format(repr(type(f).__name__)))
        self.f = f

    def __call__(self, x):
        (a, b) = x
        return self.f(a), self.f(b)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Inspect:
    """
    Print the current value then return it
    """

    def __call__(self, x):
        print(x)
        return x


if __name__ == '__main__':
    input_value = 7
    transform = Compose([
        Tupled(3),
        First(lambda x: x + 1),
        Second(lambda x: x - 1),
        MapAll(lambda x: x * 2),
        Inspect(),
        First(lambda x: x // 3),
        Second(lambda x: x * 2),
        Third(lambda x: x + 1),
        Inspect(),
        MapAll(str),
        Lambda(lambda x: ''.join(x))
    ])

    output = transform(input_value)
    print(output)
    assert output == '52415'
