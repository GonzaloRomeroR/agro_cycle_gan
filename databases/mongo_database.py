from .database import DatabaseClient
from pymongo import MongoClient


class MongoDatabaseClient(DatabaseClient):
    """
    Client for mongo database connections
    """

    def __init__(self) -> None:
        super().__init__()
        self.client = None
        self.db = None
        self.collection = None

    def connect(self, connection_string: str) -> None:
        self.client = MongoClient(connection_string)
        print(self.client)

    def create_database(self, name: str):
        self.select_database(name)
        return self.client[name]

    def delete_database(self, name: str) -> None:
        self.client.drop_database(name)

    def select_database(self, name: str) -> None:
        self.db = self.client[name]

    def database_exists(self, name: str) -> bool:
        return name not in self.db_client.list_database_names()

    def select_collection(self, name: str) -> None:
        if self.db is None:
            print("No db selected")
            return
        self.collection = self.db[name]

    def collection_insert(self, json: str) -> None:
        if self.db is None or self.collection is None:
            print("No db or collection selected")
            return
        self.collection.insert_many([json])


if __name__ == "__main__":
    pass
