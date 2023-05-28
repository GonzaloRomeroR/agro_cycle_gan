from .mongo_database import MongoDatabaseClient
from typing import Dict, Any


class DBHandler:
    def __init__(self, connection_string: str, db_type: str = "mongo") -> None:
        self.db_client = self.create_db(db_type)
        self.db_client.connect(connection_string)

    def create_db(self, name: str) -> None:
        """
        Database factory
        """
        if name == "mongo":
            return MongoDatabaseClient()
        else:
            return MongoDatabaseClient()  # Only database available so far

    def add_to_database(
        self, db_name: str, collection: str, json: Dict[str, Any]
    ) -> None:
        """
        Add information to database
        """

        if not self.db_client.database_exists:
            self.db_client.create_database(db_name)

        self.db_client.select_database(db_name)

        self.db_client.select_collection(collection)
        self.db_client.collection_insert(json)
