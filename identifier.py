from typing import Generic, TypeVar

T = TypeVar("T")


class Identifier(Generic[T]):
    """Maps objects to IDs and back. Requires T to not be an int."""

    def __init__(self):
        # id -> obj
        self.__obj: dict[int, T] = {}
        # obj -> id
        self.id: dict[T, int] = {}
        self.__next_id = 0

    def add(self, obj: T):
        """Assigns an id to the object (if not already assigned)."""

        # if an id was already assigned, do nothing
        if obj in self.id:
            return
        # assign id
        self.id[obj] = self.__next_id
        self.__obj[self.__next_id] = obj
        self.__next_id += 1

    def __contains__(self, key):
        """If object or id is contained."""

        if type(key) == int:
            return key in self.__obj
        return key in self.id

    def __getitem__(self, key) -> T:
        """Gets object by id."""

        return self.__obj[key]

    def __delitem__(self, key):
        """Deletes object by id."""

        obj = self.__obj[key]
        del self.id[obj]
        del self.__obj[key]
