from abc import abstractmethod, ABC


class DatabaseClient(ABC):
    """
    Interface to manage databases
    """

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def create_database(self, name: str):
        pass

    @abstractmethod
    def delete_database(self, name: str):
        pass

    @abstractmethod
    def select_database(self, name: str):
        pass

    @abstractmethod
    def database_exists(self, name: str) -> bool:
        pass
