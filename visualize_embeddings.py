import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import json
import math
import re # Import re for regular expressions

def extract_message_and_type_from_string(text_string):
    """
    Extracts message and type from a string that resembles a dictionary or JSON object.
    Handles a potential 'User: ' prefix.

    Args:
        text_string (str): The input string.

    Returns:
        tuple: (message, type)
               message defaults to a truncated version of the input string if not found.
               type defaults to 'unknown' if not found.
    """
    if not isinstance(text_string, str):
        text_string = str(text_string) # Ensure it's a string

    # Default values
    message = text_string[:70] + '...' if len(text_string) > 70 else text_string # Default message
    msg_type = "unknown"

    effective_string = text_string
    if text_string.startswith("User: "):
        effective_string = text_string[len("User: "):]

    # Regex to find "key": "value" (handles escaped quotes in value)
    def find_value(key, s):
        # Pattern: "key" followed by optional whitespace, then colon, then optional whitespace, 
        # then opening quote for value, then captured group for value content, then closing quote.
        # Value content (group 1) can be anything NOT a quote, OR an escaped quote.
        match = re.search(r'"{}"\s*:\s*"((?:\\"|[^\"])*)"'.format(re.escape(key)), s, re.IGNORECASE)
        if match:
            return match.group(1).replace('\\"', '"').strip() # Unescape and strip
        return None

    extracted_type = find_value("type", effective_string)
    if extracted_type is not None:
        msg_type = extracted_type

    # Prioritize "user_message", then "message"
    user_message_content = find_value("user_message", effective_string)
    if user_message_content is not None:
        message_content = user_message_content
    else:
        message_content = find_value("message", effective_string)
    
    if message_content is not None:
        # Clean up escaped newlines/tabs from the extracted message string
        cleaned_message = message_content.replace('\\n', ' ').replace('\\t', ' ').strip()
        message = cleaned_message[:50] + '...' if len(cleaned_message) > 50 else cleaned_message
    
    # print(f"Extracted from string '{text_string[:30]}...': msg='{message}', type='{msg_type}'")
    return message, msg_type

def load_embeddings(directory_path):
    """
    Load all embedding files from the specified directory and its subdirectories.
    
    Args:
        directory_path (str): Path to the directory containing embedding files
        
    Returns:
        dict: Dictionary mapping subdirectories to lists of tuples (filename, embedding_vector, label_text)
    """
    print(f"Loading embeddings from {directory_path}")
    
    embeddings_by_subdir = {}
    
    # Walk through directory and all subdirectories
    for root, dirs, files in os.walk(directory_path):
        print(f"Processing directory: {root}")
        # Get all .npy files in the current directory
        npy_files = glob.glob(os.path.join(root, "*.npy"))
        
        if npy_files:
            print(f"Found {len(npy_files)} .npy files in {root}")
        
        # Create a key for this subdirectory
        subdir = os.path.relpath(root, directory_path)
        if subdir not in embeddings_by_subdir:
            embeddings_by_subdir[subdir] = []
        
        # Load each .npy embedding file and its corresponding JSON file
        for file_path in npy_files:
            try:
                # Load embedding
                embedding = np.load(file_path)
                print(f"Embedding from {file_path}:")
                print(f"  Shape: {embedding.shape}")
                print(f"  Type: {embedding.dtype}")
                print(f"  First few values: {embedding[:5] if embedding.size > 5 else embedding}")
                
                filename = os.path.basename(file_path)  # Just use the filename for better readability
                
                # Get the corresponding JSON file path
                json_file_path = os.path.join(os.path.dirname(file_path), os.path.splitext(filename)[0] + '.json')
                
                label_text = "No content extracted"
                message_type = "unknown" # Default message type

                if os.path.exists(json_file_path):
                    try:
                        with open(json_file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        # Attempt 1: Extract from top-level keys of the loaded JSON object
                        if isinstance(json_data, dict):
                            # Use .get() to safely access keys, providing current vals as defaults
                            raw_message = json_data.get("message", label_text)
                            # If raw_message itself is a dict (e.g. full inner message), stringify for consistent processing by extract_ fn
                            if isinstance(raw_message, dict):
                                raw_message = str(raw_message)

                            raw_type = json_data.get("type", message_type)
                            if isinstance(raw_type, dict):
                                raw_type = str(raw_type)

                            # Initial extraction from top-level
                            label_text, message_type = extract_message_and_type_from_string(raw_message)
                            # If type was more definitively found at top-level, keep it.
                            if raw_type != "unknown" and raw_type != message_type : # if top-level had a type, and extract_ didn't find a better one from raw_message
                                 # Check if extract_ found something from raw_message. If extract_ found unknown, top-level wins.
                                 extracted_type_from_raw_msg = extract_message_and_type_from_string(raw_message)[1]
                                 if extracted_type_from_raw_msg == "unknown":
                                     message_type = raw_type

                        # Attempt 2: If json_data has a "text" field, parse it using our function
                        # This might refine or override what was found at the top level.
                        if isinstance(json_data, dict) and "text" in json_data:
                            text_field_content = json_data["text"]
                            # The extract_message_and_type_from_string handles string conversion and parsing
                            extracted_label, extracted_type = extract_message_and_type_from_string(text_field_content)
                            
                            # Update if the new extraction is more specific or different from defaults
                            if extracted_label != "No content extracted" and not extracted_label.startswith(str(text_field_content)[:50]):
                                label_text = extracted_label
                            if extracted_type != "unknown":
                                message_type = extracted_type
                        
                        # Fallback if label_text is still a placeholder but json_data was loaded (e.g. json_data was a list)
                        if (label_text == "No content extracted" or label_text.startswith(str(json_data)[:50])) and isinstance(json_data, (list, str)):
                            label_text, message_type = extract_message_and_type_from_string(str(json_data))

                    except json.JSONDecodeError as e_json_decode:
                        print(f"Error decoding JSON file {json_file_path}: {e_json_decode}")
                        # Try to extract from the raw file content if it failed to parse as JSON
                        try:
                            with open(json_file_path, 'r', encoding='utf-8') as f_raw:
                                raw_content_str = f_raw.read()
                            label_text, message_type = extract_message_and_type_from_string(raw_content_str)
                            if label_text.startswith(raw_content_str[:50]): # if still just raw content
                                label_text = "JSON Decode Error"
                        except Exception as e_raw_read:
                            print(f"Could not read raw {json_file_path} after JSON decode error: {e_raw_read}")
                            label_text = "JSON Decode Error"
                            message_type = "error"

                    except Exception as e_json_other:
                        print(f"Error processing JSON file {json_file_path}: {e_json_other}")
                        label_text = "JSON Process Error"
                        message_type = "error"
                else:
                    label_text = "No JSON File"
                    message_type = "no_file"
                
                # Final truncation is handled by extract_message_and_type_from_string or done here if needed
                # The function already truncates, so this is a safeguard or for non-string cases.
                if not isinstance(label_text, str):
                    final_label = str(label_text)[:50] + '...' if len(str(label_text)) > 50 else str(label_text)
                elif len(label_text) > 53 and label_text.endswith("..."):
                    final_label = label_text # Already truncated by extract function
                else:
                    final_label = label_text[:50] + '...' if len(label_text) > 50 else label_text

                embeddings_by_subdir[subdir].append((filename, embedding, final_label, message_type))
                
                # Print contents of corresponding text file if it exists (for debugging)
                txt_file_path = os.path.join(os.path.dirname(file_path), os.path.splitext(filename)[0] + '.txt')
                if os.path.exists(txt_file_path):
                    try:
                        with open(txt_file_path, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                            print(f"Contents of {txt_file_path}:")
                            print(text_content)
                            print("-" * 80)  # Separator for readability
                    except Exception as e:
                        print(f"Error reading text file {txt_file_path}: {e}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
    # Remove empty subdirectories
    embeddings_by_subdir = {k: v for k, v in embeddings_by_subdir.items() if v}
    
    if not embeddings_by_subdir:
        print(f"No embedding files found in {directory_path} or its subdirectories from which data could be extracted.")
    else:
        total_embeddings = sum(len(v) for v in embeddings_by_subdir.values())
        print(f"Successfully loaded {total_embeddings} embeddings across {len(embeddings_by_subdir)} subdirectories for processing")
    
    return embeddings_by_subdir

def visualize_embeddings_2d(embeddings_by_subdir, color_map, default_color, method='pca', perplexity=30):
    """
    Visualize embeddings in 2D using dimensionality reduction with a subplot for each subdirectory.
    
    Args:
        embeddings_by_subdir (dict): Dictionary mapping subdirectories to lists of tuples 
                                     (filename, embedding_vector, label_text, message_type)
        color_map (dict): Mapping from message_type to color string.
        default_color (str): Default color string for types not in color_map.
        method (str): Dimensionality reduction method ('pca' or 'tsne')
        perplexity (int): Perplexity parameter for t-SNE
    """
    if not embeddings_by_subdir:
        print("No embeddings to visualize")
        return
    
    # Set the Seaborn style
    sns.set(style="whitegrid")
    
    # Use a modern color palette
    palette = sns.color_palette("viridis", n_colors=len(color_map))
    color_map_updated = {k: palette[i] for i, k in enumerate(color_map.keys())}
    default_color = sns.color_palette("Set2")[0]  # A distinct color for default
    
    # Calculate grid dimensions for subplots
    n_subdirs = len(embeddings_by_subdir)
    n_cols = min(3, n_subdirs)  # Maximum 3 columns
    n_rows = math.ceil(n_subdirs / n_cols)
    
    # Create figure and subplots with improved aesthetics
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows), dpi=100, facecolor='white')
    
    # Method name for title
    method_name = "PCA" if method.lower() == 'pca' else f"t-SNE (perplexity={perplexity})"
    
    # Create a subplot for each subdirectory
    for i, (subdir, embeddings) in enumerate(embeddings_by_subdir.items()):
        # Extract vectors and labels for this subdirectory
        filenames = [item[0] for item in embeddings]
        labels = [item[2] for item in embeddings]  # Get pre-loaded labels
        
        print(f"\nProcessing subdirectory for visualization: {subdir}")
        print(f"Number of embeddings in this subdir: {len(embeddings)}")
        
        # Temporarily store vectors to check shapes before creating NumPy array
        temp_vectors = []
        temp_labels = []
        temp_filenames = []
        temp_types = []

        for k, item_tuple in enumerate(embeddings):
            emb_vector = item_tuple[1]
            # Basic check if it's list-like and its elements are numbers
            is_valid_vector = True
            # Check for NumPy array emptiness
            if not hasattr(emb_vector, '__len__') or (hasattr(emb_vector, 'size') and emb_vector.size == 0) or (not hasattr(emb_vector, 'size') and not emb_vector):
                is_valid_vector = False
            elif not all(isinstance(x, (int, float, np.number)) for x in emb_vector): # Check if all elements are numbers
                is_valid_vector = False 

            if is_valid_vector:
                print(f"  Embedding {k} ({filenames[k]}): shape {np.shape(emb_vector)}, type {type(emb_vector)}")
                temp_vectors.append(emb_vector)
                temp_labels.append(labels[k])
                temp_filenames.append(filenames[k])
                temp_types.append(embeddings[k][3])
            else:
                print(f"  Skipping problematic embedding {k} ({filenames[k]}) in subdir {subdir}. Vector: {str(emb_vector)[:100]}...")
        
        if not temp_vectors:
            print(f"Skipping {subdir}: no valid vectors after filtering.")
            continue
        
        # Check for consistent lengths before creating np.array
        first_vector_len = -1
        if temp_vectors:
            first_vector_len = len(temp_vectors[0])
        
        consistent_vectors = []
        consistent_labels = []
        consistent_types = []
        consistent_filenames = []

        for idx, vec in enumerate(temp_vectors):
            if len(vec) == first_vector_len:
                consistent_vectors.append(vec)
                consistent_labels.append(temp_labels[idx])
                consistent_types.append(temp_types[idx])
                consistent_filenames.append(temp_filenames[idx])
            else:
                print(f"  WARNING: Inconsistent vector length in {subdir} for {temp_filenames[idx]}. Expected {first_vector_len}, got {len(vec)}. Skipping.")

        if not consistent_vectors:
            print(f"Skipping {subdir}: no consistently shaped vectors after filtering.")
            continue
        
        vectors = np.array(consistent_vectors)
        current_labels = consistent_labels
        current_types = consistent_types

        if len(vectors) < 2:
            print(f"Skipping {subdir}: not enough consistently shaped vectors for dimensionality reduction (need at least 2, got {len(vectors)})")
            continue
    
        # Reduce dimensionality
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            reduced_vectors = reducer.fit_transform(vectors)
        elif method.lower() == 'tsne':
            effective_perplexity = min(perplexity, len(vectors) - 1)
            if effective_perplexity <= 0:
                print(f"Skipping {subdir} for t-SNE: not enough samples for perplexity {perplexity} (samples: {len(vectors)})")
                continue # This continue is for the main loop over subdirectories
            reducer = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42, init='pca', learning_rate='auto')
            reduced_vectors = reducer.fit_transform(vectors)
        else: # This else aligns with the if/elif for dimensionality reduction method
            print(f"Unknown method: {method}. Using PCA.")
            reducer = PCA(n_components=2)
            reduced_vectors = reducer.fit_transform(vectors)
        
        # Calculate centroid and find the point closest to it
        centroid_2d = np.mean(reduced_vectors, axis=0)
        distances_to_centroid = np.linalg.norm(reduced_vectors - centroid_2d, axis=1)
        closest_point_idx = np.argmin(distances_to_centroid)
        furthest_point_idx = np.argmax(distances_to_centroid) # Find the furthest point

        # Determine radius for the circle: distance to the second furthest point
        # or furthest if only 2 points, or a small default if only 1 point.
        circle_radius = 0
        if len(distances_to_centroid) > 0:
            sorted_distances = np.sort(distances_to_centroid)[::-1] # Sort descending
            if len(sorted_distances) >= 2:
                circle_radius = sorted_distances[1] # Second furthest
            elif len(sorted_distances) == 1:
                 circle_radius = sorted_distances[0] # Furthest (if only one point, this is its distance to itself, so 0, effectively no circle or tiny)
        
        # Determine colors for each point with improved aesthetics
        point_colors = []
        for idx, msg_type in enumerate(current_types): 
            if idx == closest_point_idx:
                point_colors.append(sns.color_palette("husl", 8)[2])  # Bright green for centroid point
            elif idx == furthest_point_idx:
                point_colors.append(sns.color_palette("husl", 8)[0])  # Bright red for furthest point
            else:
                point_colors.append(color_map_updated.get(msg_type, default_color))
        
        # Create subplot with improved aesthetics
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Plot points with improved aesthetics
        scatter = ax.scatter(
            reduced_vectors[:, 0], 
            reduced_vectors[:, 1], 
            alpha=0.8, 
            c=point_colors,
            s=80,  # Larger point size
            edgecolor='white',  # White edge for better visibility
            linewidth=0.5
        )
        
        # Plot the actual centroid with improved aesthetics
        ax.scatter(
            centroid_2d[0], 
            centroid_2d[1], 
            marker='X', 
            color='black', 
            s=150,  # Larger size for emphasis
            zorder=5,  # Ensure it's on top
            edgecolor='white',
            linewidth=1.0
        )

        # Draw the circle with improved aesthetics
        if circle_radius > 0.001:  # Avoid drawing tiny or zero-radius circles
            circle_patch = plt.Circle(
                (centroid_2d[0], centroid_2d[1]), 
                circle_radius, 
                fill=False, 
                color='gray', 
                linestyle='dashed',  # Dashed instead of dotted for better visibility
                linewidth=1.5,
                alpha=0.7
            )
            ax.add_patch(circle_patch)

        # Add labels for points with improved aesthetics
        for j, label_text in enumerate(current_labels):
            # Extract just the message part if it's in the format of a dictionary string
            try:
                # Try to parse as a dictionary string
                if label_text.startswith('{') and label_text.endswith('}'):
                    label_dict = eval(label_text)
                    if isinstance(label_dict, dict) and 'message' in label_dict:
                        message = label_dict['message']
                        # Truncate if too long
                        label_text = message[:50] + '...' if len(message) > 50 else message
            except:
                # If parsing fails, keep the original label_text
                pass
                
            # Add text with better positioning and styling
            ax.annotate(
                label_text, 
                (reduced_vectors[j, 0], reduced_vectors[j, 1]),
                fontsize=8,
                alpha=0.9,
                xytext=(5, 5),  # Offset text slightly from point
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
            )
        
        # Set subplot title and labels with improved styling
        subdir_title = "Root" if subdir == "." else subdir
        ax.set_title(f"{len(vectors)} embeddings", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("PCA Component 1", fontsize=12, labelpad=10)
        ax.set_ylabel("PCA Component 2", fontsize=12, labelpad=10)
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a light background color to distinguish the plot area
        ax.set_facecolor('#f8f9fa')
    
    # Set overall title with improved styling
    fig.suptitle(
        f"{method_name} Visualization of Embeddings by Subdirectory", 
        fontsize=18, 
        fontweight='bold', 
        y=0.98
    )
    
    # Add a legend for message types
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map_updated.get(t, default_color), 
                          markersize=10, label=t) for t in color_map.keys()]
    # Add special markers for centroid and furthest point
    handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black', 
                             markersize=10, label='Centroid'))
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sns.color_palette("husl", 8)[0], 
                             markersize=10, label='Furthest Point'))
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sns.color_palette("husl", 8)[2], 
                             markersize=10, label='Closest to Centroid'))
    
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.01), 
               frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout with more space for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure with high quality
    plt.savefig(f"embedding_visualization_{method.lower()}.png", dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def main():
    # Directory containing embedding files
    embedding_dir = "/home/samer/Documents/LAU/Research/focus_memgpt/agent_step_embedding_logs"
    
    # Color mapping for types
    current_color_map = {
        "user_message": "black",
        # "login": "grey",
        # "unknown": "blue", # Explicitly define for unknown if needed, or rely on default_color
        # "no_file": "orange",
        "error": "red"
        # Add other types and their colors here
    }
    current_default_color = "purple" # Default color for types not in color_map
    
    # Load embeddings
    embeddings_by_subdir = load_embeddings(embedding_dir)
    
    if embeddings_by_subdir:
        # Filter for embeddings of type "user_message" before visualization
        filtered_embeddings_by_subdir = {}
        for subdir, embedding_list in embeddings_by_subdir.items():
            user_message_embeddings = [
                emb for emb in embedding_list if emb[3] == "user_message" # emb[3] is message_type
            ]
            if user_message_embeddings: # Only add subdir if it has user messages
                filtered_embeddings_by_subdir[subdir] = user_message_embeddings
        
        if not filtered_embeddings_by_subdir:
            print("No embeddings of type 'user_message' found to visualize.")
            return # Exit if nothing to plot

        # The print below should reflect the count of user_message embeddings
        total_filtered_embeddings = sum(len(v) for v in filtered_embeddings_by_subdir.values())
        print(f"Visualizing {total_filtered_embeddings} embeddings of type 'user_message' across {len(filtered_embeddings_by_subdir)} subdirectories.")
        
        # Visualize using PCA
        visualize_embeddings_2d(filtered_embeddings_by_subdir, current_color_map, current_default_color, method='pca')
        
        # # Visualize using t-SNE
        # visualize_embeddings_2d(filtered_embeddings_by_subdir, current_color_map, current_default_color, method='tsne', perplexity=30)
    else:
        print("No embeddings were loaded. Please check the directory path.")

if __name__ == "__main__":
    main()
