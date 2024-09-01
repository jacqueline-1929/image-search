import os
import time
import torch
import torchvision
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging

# Set up logging
logging.basicConfig(filename='image_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print(f"PyTorch version: {torch.__version__}")
logging.info(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
logging.info(f"Torchvision version: {torchvision.__version__}")

print("Script started")
logging.info("Script started")
print(f"Current working directory: {os.getcwd()}")
logging.info(f"Current working directory: {os.getcwd()}")

print(f"CUDA available: {torch.cuda.is_available()}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logging.info(f"Using device: {device}")

print("Loading MobileNetV2 model...")
logging.info("Loading MobileNetV2 model...")
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model = model.to(device).eval()
print("MobileNetV2 loaded and set to eval mode")
logging.info("MobileNetV2 loaded and set to eval mode")

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
logging.info("Connected to Milvus")

def create_milvus_collection(collection_name, dim=1280):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logging.info(f"Dropped existing collection: {collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="feature_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="margin", dtype=DataType.FLOAT),
        FieldSchema(name="inventory", dtype=DataType.INT64)
    ]
    schema = CollectionSchema(fields, "Image features and product data")
    collection = Collection(collection_name, schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="feature_vector", index_params=index_params)
    logging.info(f"Created Milvus collection: {collection_name}")
    return collection

def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256), Image.LANCZOS)
    
    width, height = img.size
    left = (width - target_size[0]) // 2
    top = (height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    img = img.crop((left, top, right, bottom))
    
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    
    return img_tensor

def extract_features(img_path):
    img_tensor = preprocess_image(img_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)
    
    features = features.squeeze().cpu().numpy()
    
    # Ensure the feature vector has 1280 dimensions
    if len(features) != 1280:
        logging.warning(f"Unexpected feature dimension for {img_path}. Expected 1280, got {len(features)}.")
        features = np.zeros(1280)  # Return a zero vector if dimensions don't match
    
    return features

def process_images(archive_dir, collection):
    logging.info(f"Processing images in directory: {archive_dir}")
    print(f"Processing images in directory: {archive_dir}")
    batch_size = 100  # Reduced batch size
    entities = []
    total_processed = 0
    
    for root, _, files in os.walk(archive_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    features = extract_features(img_path)
                    
                    # Ensure features have the correct dimensionality
                    if len(features) != 1280:
                        logging.warning(f"Skipping {img_path} due to incorrect feature dimension.")
                        continue
                    
                    relative_path = os.path.relpath(img_path, archive_dir)
                    
                    # Generate dummy data for margin and inventory
                    margin = np.random.uniform(0.1, 0.5)  # 10% to 50% margin
                    inventory = np.random.randint(0, 100)  # 0 to 100 items in stock
                    
                    entities.append({
                        "id": total_processed + len(entities) + 1,
                        "feature_vector": features.tolist(),  # Ensure it's a list
                        "image_path": relative_path,
                        "margin": float(margin),
                        "inventory": int(inventory)
                    })
                    logging.info(f"Processed: {relative_path}")
                    print(f"Processed: {relative_path}")

                    # Insert in smaller batches
                    if len(entities) >= batch_size:
                        try:
                            collection.insert(entities)
                            logging.info(f"Inserted batch of {len(entities)} entities into Milvus")
                            print(f"Inserted batch of {len(entities)} entities into Milvus")
                            total_processed += len(entities)
                            entities = []
                        except Exception as e:
                            logging.error(f"Error inserting batch: {str(e)}")
                            print(f"Error inserting batch: {str(e)}")
                            entities = []  # Clear the batch on error

                except Exception as e:
                    logging.error(f"Error processing {img_path}: {str(e)}")
                    print(f"Error processing {img_path}: {str(e)}")
    
    # Insert any remaining entities
    if entities:
        try:
            collection.insert(entities)
            logging.info(f"Inserted final batch of {len(entities)} entities into Milvus")
            print(f"Inserted final batch of {len(entities)} entities into Milvus")
            total_processed += len(entities)
        except Exception as e:
            logging.error(f"Error inserting final batch: {str(e)}")
            print(f"Error inserting final batch: {str(e)}")

    collection.flush()
    logging.info(f"Total entities inserted into Milvus: {total_processed}")
    logging.info(f"Total entities in collection: {collection.num_entities}")
    print(f"Total entities inserted into Milvus: {total_processed}")
    print(f"Total entities in collection: {collection.num_entities}")

def search_similar_products(collection, query_vector, top_k=5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="feature_vector",
        param=search_params,
        limit=top_k,
        output_fields=["image_path", "margin", "inventory"]
    )
    return results[0]

def calculate_score(distance, margin, inventory):
    similarity_score = 1 / (1 + distance)  # Convert distance to similarity
    inventory_score = min(1, inventory / 100)  # Normalize inventory score
    
    # Adjust weights based on your business priorities
    return (0.5 * similarity_score) + (0.3 * margin) + (0.2 * inventory_score)

if __name__ == "__main__":
    archive_dir = 'Archive'
    collection_name = "product_images"
    
    logging.info(f"Archive directory set to: {archive_dir}")
    print(f"Archive directory set to: {archive_dir}")

    if not os.path.exists(archive_dir):
        logging.error(f"Error: Directory {archive_dir} does not exist.")
        print(f"Error: Directory {archive_dir} does not exist.")
    else:
        # Create Milvus collection
        collection = create_milvus_collection(collection_name, dim=1280)  # MobileNetV2 output dimension
        
        # Process images and insert into Milvus
        process_images(archive_dir, collection)
        
        # Load the collection before querying
        collection.load()
        logging.info("Collection loaded into memory")
        print("Collection loaded into memory")
        
        # Perform similarity search
        logging.info("\nTesting search performance...")
        print("\nTesting search performance...")
        num_queries = min(5, collection.num_entities)
        for i in range(num_queries):
            try:
                query_image = collection.query(expr=f"id == {i+1}", output_fields=["feature_vector", "image_path"])
                if query_image:
                    query_vector = query_image[0]['feature_vector']
                    logging.info(f"\nQuerying with image: {query_image[0]['image_path']}")
                    print(f"\nQuerying with image: {query_image[0]['image_path']}")

                    start_time = time.time()
                    results = search_similar_products(collection, query_vector)
                    search_time = time.time() - start_time
                    logging.info(f"Search time: {search_time:.4f} seconds")
                    print(f"Search time: {search_time:.4f} seconds")
                    
                    logging.info("Results:")
                    print("Results:")
                    for hit in results:
                        score = calculate_score(hit.distance, hit.entity.get('margin'), hit.entity.get('inventory'))
                        result_str = f"  {hit.entity.get('image_path')}: " \
                                     f"distance {hit.distance:.4f}, " \
                                     f"margin {hit.entity.get('margin'):.2f}, " \
                                     f"inventory {hit.entity.get('inventory')}, " \
                                     f"score {score:.4f}"
                        logging.info(result_str)
                        print(result_str)
                else:
                    logging.warning(f"No image found for id {i+1}")
                    print(f"No image found for id {i+1}")
            except Exception as e:
                logging.error(f"Error during search for id {i+1}: {str(e)}")
                print(f"Error during search for id {i+1}: {str(e)}")

        # Release the collection after searching
        collection.release()
        logging.info("Collection released from memory")
        print("Collection released from memory")

    logging.info("Script finished")
    print("Script finished")
