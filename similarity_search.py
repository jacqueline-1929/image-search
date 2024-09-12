import os
import time
import torch
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("Script started")
print(f"Current working directory: {os.getcwd()}")

print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading MobileNetV2 model...")
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model = model.to(device).eval()
print("MobileNetV2 loaded and set to eval mode")

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
print("Connected to Milvus")

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
    
    return features.squeeze().cpu().numpy()

def create_milvus_collection(collection_name, dim=1000):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="feature_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500)
    ]
    schema = CollectionSchema(fields, "Image features collection")
    collection = Collection(collection_name, schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="feature_vector", index_params=index_params)
    print(f"Created Milvus collection: {collection_name}")
    return collection

def process_images(archive_dir, collection):
    print(f"Processing images in directory: {archive_dir}")
    total_processed = 0
    batch_size = 100
    entities = []

    for root, _, files in os.walk(archive_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    features = extract_features(img_path)
                    relative_path = os.path.relpath(img_path, archive_dir)
                    
                    entities.append({
                        "id": total_processed,
                        "feature_vector": features.tolist(),
                        "image_path": relative_path
                    })
                    print(f"Processed: {relative_path}")
                    
                    total_processed += 1
                    
                    if len(entities) >= batch_size:
                        collection.insert(entities)
                        entities = []
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    if entities:
        collection.insert(entities)
    
    collection.flush()
    print(f"Total entities inserted into Milvus: {total_processed}")
    print(f"Total entities in collection: {collection.num_entities}")

def calculate_similarity_percentage(distance, max_distance=1000):
    """
    Calculate similarity percentage based on distance.
    A distance of 0 is 100% similar, and distances >= max_distance are 0% similar.
    """
    similarity = max(0, 1 - (distance / max_distance))
    return similarity * 100

def search_similar_images(collection, query_vector, k=5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="feature_vector",
        param=search_params,
        limit=k,
        output_fields=["image_path"]
    )
    return results[0]

if __name__ == "__main__":
    archive_dir = 'Archive'
    collection_name = "image_features"
    print(f"Archive directory set to: {archive_dir}")

    if not os.path.exists(archive_dir):
        print(f"Error: Directory {archive_dir} does not exist.")
    else:
        # Create Milvus collection
        collection = create_milvus_collection(collection_name, dim=1000)
        
        # Process images and insert into Milvus
        process_images(archive_dir, collection)
        
        # Load the collection before querying
        collection.load()
        print("Collection loaded into memory")
        
        print("\nTesting search performance...")
        num_queries = min(5, collection.num_entities)
        for i in range(num_queries):
            try:
                query_image = collection.query(expr=f"id == {i}", output_fields=["feature_vector", "image_path"])
                if query_image:
                    query_vector = query_image[0]['feature_vector']
                    print(f"\nQuerying with image: {query_image[0]['image_path']}")

                    start_time = time.time()
                    results = search_similar_images(collection, query_vector)
                    search_time = time.time() - start_time
                    print(f"Search time: {search_time:.4f} seconds")
                    
                    print("Results:")
                    for hit in results:
                        similarity = calculate_similarity_percentage(hit.distance)
                        print(f"  {hit.entity.get('image_path')}: "
                              f"distance {hit.distance:.4f}, "
                              f"similarity {similarity:.2f}%")
                else:
                    print(f"No image found for id {i}")
            except Exception as e:
                print(f"Error during search for id {i}: {str(e)}")

        # Release the collection after searching
        collection.release()
        print("Collection released from memory")

    print("Script finished")
