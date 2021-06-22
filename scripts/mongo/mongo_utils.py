"""
Contains functions for interacting with MongoDB database
"""

import pprint
from pymongo import MongoClient
from typing import List, Tuple

# Create a MongoDB client
client = MongoClient()

# Connect to our specific MongoDB instance
client = MongoClient('localhost', 27017)

# Connect to the mbusi_test database
db = client.mbusi_test


def print_collection(collection: any, collection_name: str) -> None:
    """Prints all documents in a given MongoDB collection"""
    print(f'{collection_name} collection: ')
    for doc in collection.find():
        pprint.pprint(doc)


def insert_batch_into_database(detections: List[Tuple[List[str], str]], drop_collection: bool = False) -> None:
    """
    Inserts detections from a batch of images into the database.

    Parameters
    ----------
    detections: The detected text from the batch.
        Each element in the detections list is a tuple, where the first element is a list of
        detected strings of text, and the second element is the filename where that string of
        text was found.
    drop_collection: True if we want to drop the collection after inserting into it.
        Only need this while testing, so as not to store anything in the MongoDB instance.

    Returns
    -------
    None

    """
    # Loop over all detections
    for detected_strings, image_name in detections:

        # Create a new dictionary (which will be sent to MongoDB as a JSON object)
        new_label = {
            "image_name": image_name,
            "detected_strings": detected_strings
        }

        # Add the label into the database
        new_label_id = db.labels.insert_one(new_label)

    # Print all of the documents in the labels ollection
    print_collection(db.labels, "labels")

    # Drop the collection if we're just testing things
    if drop_collection:
        db.labels.drop()


def test_mongo():
    """Dummy script to test inserting things into a MongoDB instance"""
    # Get the persons collection from our database
    persons = db.persons

    # Print all of the documents in the persons collection
    print("Previous persons collection")
    for doc in persons.find():
        pprint.pprint(doc)

    # Create a new person
    new_person = {"age": 21,
                  "hair_color": "brown",
                  "location": "Nashville",
                  "name": "sam"}

    # Set our query and update
    my_query = {"name": new_person["name"]}
    my_update = {"$set": new_person}

    # Update the person, inserting a new one if the person doesn't exist
    new_person_id = persons.insert_one(filter=my_query, update=my_update, upsert=True)

    # Print all of the documents in the persons collection
    print("Updated persons collection")
    for doc in db.persons.find():
        pprint.pprint(doc)

    # Drop the collection
    persons.drop()


if __name__ == "__main__":
    test_mongo()
