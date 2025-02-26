from firebase_config import db

def test_firebase_connection():
    try:
        # Try to access a collection
        test_collection = db.collection('test')
        print("Successfully connected to Firestore!")
        
        # Try to write some test data
        doc_ref = test_collection.document('test_doc')
        doc_ref.set({
            'test': 'data',
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print("Successfully wrote to Firestore!")
        
        # Try to read the data back
        doc = doc_ref.get()
        print(f"Successfully read from Firestore: {doc.to_dict()}")
        
        # Clean up
        doc_ref.delete()
        print("Successfully deleted test document")
        
    except Exception as e:
        print(f"Error testing Firebase connection: {str(e)}")
        raise

if __name__ == "__main__":
    test_firebase_connection() 