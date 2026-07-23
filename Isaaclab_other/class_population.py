import uuid
import json
import os


class Lineage:
    def __init__(self):
        # lineage structure where each individual is accessed by (generation, individual number)
        self.lineage = {}

    def add_individual(self, parent_generation, parent_number, urdf_info, task_score, new_id, tag="active", metadata=None):
        """
        指定某个父辈来添加个体
        Adds a new individual to the lineage, based on the parent generation and number.
        The new individual is given a unique number based on its parent's generation.

        :param parent_generation: (int) The generation number of the parent.
        :param parent_number: (int) The individual number of the parent.
        :param urdf_info: (dict) The URDF information of the new individual.
        :param task_score: (dict) The task scoring information for the new individual.
        :param tag: (str) A tag indicating if the individual is active or eliminated (default: "active").
        :param new_id: (str) A unique identifier for the new individual.
        """
        # Generate a unique individual number for this new individual
        # Find the maximum individual number from the parent generation or start with 0
        new_individual_number = max(
            [key[1] for key in self.lineage.keys() if key[0] == parent_generation+1], default=-1) + 1

        # Create the new individual's unique ID
        #new_id = uuid.uuid4().hex
        print("parent_generation :",parent_generation )
        
        # Store the new individual under the appropriate generation and number
        self.lineage[(parent_generation + 1, new_individual_number)] = {
            'parent': (parent_generation, parent_number),
            'urdf_info': urdf_info,
            'task_score': task_score,
            'tag': tag,
            'id': new_id,
            'metadata': metadata or {},
        }

        print(f"Added new individual: Generation {parent_generation + 1}, Number {new_individual_number}")

    def get_surviving_individuals_in_generation(self, generation):
        """
        Retrieves the numbers of all surviving individuals in a specific generation.

        :param generation: (int) The generation number to check.
        :return: (list) A list of individual numbers of all surviving individuals in the generation.
        """
        surviving_individuals = []

        # Iterate over all individuals in the lineage
        for (gen, individual_number), individual in self.lineage.items():
            # print("gen:",gen,generation,individual['tag'])
            if gen == generation and individual['tag'] == 'active':  # Check if individual is active
                surviving_individuals.append(individual_number)
        # print(surviving_individuals)
        return surviving_individuals

    def evaluate_and_eliminate_individuals_in_generation(self, generation, max_population):
        """
        Evaluates and eliminates individuals in a specified generation based on their task score.
        Only the top 'max_population' individuals will survive. Others will be marked as 'eliminated'.

        :param generation: (int) The generation number to evaluate.
        :param max_population: (int) The maximum number of individuals that can survive to the next generation.
        :return: (None)
        """
        # Create a list of individuals in the specified generation
        individuals_in_generation = [
            (individual_number, individual['task_score'])
            for (gen, individual_number), individual in self.lineage.items()
            if gen == generation and individual['tag'] == 'active'
        ]

        # Sort the individuals by their task score in descending order
        individuals_in_generation.sort(key=lambda x: x[1], reverse=True)

        # If the number of individuals exceeds the max_population, eliminate the lowest-scoring individuals
        if len(individuals_in_generation) > max_population:
            # Identify the cutoff score (score of the individual at position max_population)
            cutoff_score = individuals_in_generation[max_population - 1][1]

            # Eliminate individuals with a score lower than or equal to the cutoff
            for individual_number, task_score in individuals_in_generation[max_population:]:
                # Find the individual and mark them as eliminated
                for (gen, num), individual in self.lineage.items():
                    if gen == generation and num == individual_number:
                        self.lineage[(gen, num)]['tag'] = 'eliminated'
                        print(f"Individual {gen}-{num} with score {task_score} has been eliminated.")
        else:
            print(
                f"Generation {generation} has {len(individuals_in_generation)} individuals, no elimination needed.")

    def count_generations(self):
        """
        Counts the number of generations and the number of individuals in each generation.

        :return: (dict) A dictionary where keys are generations and values are the number of individuals.
        """
        generation_counts = {}
        for (generation, individual_number), _ in self.lineage.items():
            generation_counts[generation] = generation_counts.get(generation, 0) + 1

        return generation_counts

    def get_max_generation(self):
        """Return the maximum generation id in the lineage."""
        if not self.lineage:
            return -1
        return max(generation for generation, _ in self.lineage.keys())

    def has_individual_id(self, individual_id):
        """Check whether the lineage already contains an individual id."""
        return any(individual.get("id") == individual_id for individual in self.lineage.values())

    def update_individual_by_id(self, individual_id, task_score=None, metadata_updates=None, tag=None):
        """Update an existing individual in-place by its stable id."""
        for key, individual in self.lineage.items():
            if individual.get("id") != individual_id:
                continue
            if task_score is not None:
                self.lineage[key]["task_score"] = task_score
            if metadata_updates:
                merged_metadata = dict(self.lineage[key].get("metadata", {}))
                merged_metadata.update(metadata_updates)
                self.lineage[key]["metadata"] = merged_metadata
            if tag is not None:
                self.lineage[key]["tag"] = tag
            return True
        return False

    def save_to_file(self, filename):
        """
        Saves the current lineage to a file in JSON format, appending new generations to the file.

        :param filename: (str) The file name where the lineage should be saved.
        :return: (None)
        """
        print("-----------------------------")
        lineage_data = {
            "lineage": {
                f"{generation}_{individual_number}": {
                    'parent': individual['parent'],
                    'urdf_info': individual['urdf_info'],
                    'task_score': individual['task_score'],
                    'tag': individual['tag'],
                    'id': individual['id'],
                    'metadata': individual.get('metadata', {})
                }
                for (generation, individual_number), individual in self.lineage.items()
            }
        }
        print("-----------------------------")
        # print("------",filename)
        # Check if the file already exists
        if os.path.exists(filename):
            # If the file exists, load the existing data and append the new generation
            with open(filename, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    # If the file is empty or corrupted, initialize empty data
                    existing_data = {"lineage": {}}

            # Append the new generation's data to the existing data
            existing_data["lineage"].update(lineage_data["lineage"]);
            # print("existing_data:",existing_data)
            # Write the updated data back to the file
            with open(filename, 'w') as file:
                json.dump(existing_data, file, indent=4)

            print(f"Lineage has been updated in {filename}.")
        else:
            # If the file doesn't exist, create a new file and write the data
            with open(filename, 'w') as file:
                json.dump(lineage_data, file, indent=4)
            print(f"Lineage has been saved to {filename}.")


    def load_from_file(self, filename):
        """
        Loads lineage data from a JSON file and reconstructs the lineage object.

        :param filename: (str) The file name from which the lineage should be loaded.
        :return: (None)
        """
        try:
            # Read the lineage data from the file
            with open(filename, 'r') as file:
                lineage_data = json.load(file)

            # Reconstruct the lineage from the data
            for key_str, individual in lineage_data["lineage"].items():
                generation=int(key_str.split("_")  [0])
                individual_number=int(key_str.split("_")  [1])
                self.lineage[(generation, individual_number)] = {
                    'parent': individual['parent'],
                    'urdf_info': individual['urdf_info'],
                    'task_score': individual['task_score'],
                    'tag': individual['tag'],
                    'id': individual['id'],
                    'metadata': individual.get('metadata', {})
                }

            print(f"Lineage has been loaded from {filename}.")

        except FileNotFoundError:
            print(f"Error: The file {filename} does not exist.")

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from the file {filename}.")
