from pathlib import Path
import pandas as pd
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import time
import os



class LlavaInference():
    """
    LlavaInference allows us to deploy selected Llava model (locally or in NGPU - UGR, but without automation yet)
    We start with Llava1.5-7b params. It can download model, and do some inference given some images and text prompt as inputs.
    """
    def __init__(self, 
                 images: list,
                 bbdd: str,
                 classification_lvl: str,
                 n_prompt:int,
                 model:str,
                 cache: bool = True, 
                 verbose: bool = False):
        """
        Loads images from every cluster in order to do some inference on llava on ugr gpus
        Args:
            images_cluster_dict (dict)
            classification_lvl (str): Classification level to be used
            experiment_name (str): Name of the experiment for organizing results
        """

        if(model not in ("llava1-5_7b", "llava1-6_7b","llava1-6_13b")):
            raise ValueError("type must be one of followin: [llava1-5_7b, llava1-6_7b,llava1-6_13b]")
        
        # Adjust model from huggint face, but anyway, we need 2 different methods
        # depending on llava or llava-next
        if model == "llava1-5_7b":
            self.model_hf = "llava-hf/llava-1.5-7b-hf"
        elif model == "llava1-6_7b":
            self.model_hf = "llava-hf/llava-v1.6-mistral-7b-hf"
        elif model == "llava1-6_13b":
            self.model_hf = "liuhaotian/llava-v1.6-vicuna-13b"
        else:
            self.model_hf = "llava-hf/llava-v1.6-mistral-7b-hf"

        self.images = images
        self.classification_lvl = classification_lvl
        self.model = model
        self.n_prompt = n_prompt
        self.cache = cache
        self.verbose = verbose
        # Base dirs
        self.results_dir = Path(__file__).resolve().parent / f"results/{bbdd}/classification_lvl_{self.classification_lvl}/{self.model}/prompt_{self.n_prompt}"
        self.results_csv = self.results_dir / f"inference_results.csv"
        # self.classification_lvls_dir = Path(__file__).resolve().parent / "classification_lvls/"
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load categories based on classification level
        # self.categories = pd.read_csv(os.path.join(self.classification_lvls_dir, f"classification_level_{self.classification_lvl}.csv"), header=None, sep=";").iloc[:, 0].tolist()
        # categories_joins = ", ".join([category.upper() for category in self.categories])



        # self.prompt_1 = (
        #     "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and the cultural contributions of nature to people. "
        #     f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
        #     "Please adhere to the following rules:"
        #     "1. You must not assign a category that is not listed above."
        #     "2. If the image does not belong to any of the listed categories, classify it as 'NOT VALID'."
        #     "3. Provide your response exclusively as the classification, without any additional explanation or commentary."
        #     )
        

        # self.prompt_2 = (
        #     "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and the cultural contributions of nature to people. "
        #     f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
        #     "Please adhere to the following rules:"
        #     "1. You must not assign a category that is not listed above."
        #     "2. If the image does not clearly belong to any of the listed categories, classify it as the most similar category from the list."
        #     "3. If the image is not clear enough or blurry, classify it as 'NOT VALID'."
        #     "4. Provide your response EXCLUSIVELY as the classification, without any additional explanation or commentary."
        #     )
        

        # self.prompt_3 = (
        #     "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and the cultural contributions of nature to people. "
        #     f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
        #     "Please adhere to the following rules:"
        #     "1. You must not assign a category that is not listed above."
        #     "2. If the image does not clearly belong to any of the listed categories, classify it as the most similar category from the list."
        #     "3. If the image is not relevant to analyze cultural ecosystem services, classify it as NOT RELEVANT."
        #     "4. Provide your response EXCLUSIVELY as the classification, without any additional explanation or commentary."
        #     )


        # self.prompt_4 = (
        #     "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and the cultural contributions of nature to people. "
        #     f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
        #     "Please adhere to the following rules:"
        #     "1. You must not assign a category that is not listed above."
        #     "2. If the image does not clearly belong to any of the listed categories, classify it as the most similar category from the list."
        #     "3. Provide your response EXCLUSIVELY as the classification, without any additional explanation or commentary."
        #     )


        # self.prompt_5 = (
        #     "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and the cultural contributions of nature to people. "
        #     f"Your task is to classify images into TWO of the following {len(self.categories)} categories: {categories_joins}. "
        #     "Please adhere to the following rules:"
        #     "1. You must assign ONLY TWO categories from the list given above."
        #     "2. If the image does not clearly belong to any of the listed categories, classify it as the TWO most similar categories from the list."
        #     "3. DO NOT provide any explanation."
        #     "4. This is an example output: PLANTS, VEGETATION AND HABITATS."
        #     )
        

        # self.prompt_10 = (
        #     "Wrapper para clasificación con Ground Truth"
        #     )


        self.prompt_1 = """
        You are an image classification system specialized in Cultural Ecosystem Services (CES). Your task is to classify the given image into exactly one of the predefined categories below. Return only the category name and nothing else. Do not provide explanations or additional text.

        Categories:
        - Nature & Landscape: Images primarily depicting nature or landscapes, taken in wide shots.
        - Fauna & Flora: Images primarily depicting animals, plants, or parts of them, taken in close-up or medium-distance shots.
        - Recreational: Images showing the use of recreational areas, including people spending time in public recreational spaces. Shots can be close-up, medium-distance, or wide.
        - Sports: Images focused on sports activities or sports-related elements, taken in either close-up or wide shots.
        - Cultural: Images depicting cultural elements, such as traditional crafts (e.g., livestock movements, traditional weaving, basketry, or Alpujarran jarapas), taken in close-up or wide shots.
        - Religious: Images featuring religious elements (e.g., the Virgin Mary, processions, pilgrimages, churches), taken in close-up or wide shots.
        - Gastronomy: Images primarily related to gastronomy, such as dining at a restaurant or traditional food products, taken in close-up or wide shots.
        - Rural tourism: Images depicting rural tourism elements, such as rural accommodations, small villages, or countryside areas, taken in close-up, medium-distance, or wide shots.
        - Urban: Images depicting urban elements, such as houses, streets, or parks, taken in close-up or wide shots.
        - Sun & beach: Images depicting sun and beach tourism, such as people or events on the beach in a leisure context.
        - Other type: Images related to CES but not fitting any of the above categories.

        Return only the exact category name from the list above. Do not add any explanations, descriptions or additional information.
        """

        self.prompt_2 = """
        You are an image classification system specialized in Cultural Ecosystem Services (CES). Your task is to classify the given image into exactly one of the predefined categories below. Return only the category name and nothing else. Do not provide explanations or additional text.

        Categories:
        - Nature & Landscape: Images primarily depicting nature or landscapes, taken in wide shots.
        - Fauna & Flora: Images primarily depicting animals, plants, or parts of them, taken in close-up or medium-distance shots.
        - Recreational: Images showing the use of recreational areas, including people spending time in public recreational spaces (e.g., camping, people spending time together, relaxing, having fun). Shots can be close-up, medium-distance, or wide.
        - Sports: Images focused on sports activities or sports-related elements (e.g., images with ski poles, ). Shots can be taken in either close-up or wide shots.
        - Cultural: Images depicting cultural elements, such as traditional crafts (e.g., livestock movements, traditional weaving, basketry, or Alpujarran jarapas), taken in close-up or wide shots.
        - Religious: Images featuring religious elements (e.g., the Virgin Mary, processions, pilgrimages, churches), taken in close-up or wide shots.
        - Gastronomy: Images primarily related to gastronomy, such as dining at a restaurant or traditional food products, taken in close-up or wide shots.
        - Rural tourism: Images depicting rural tourism elements, such as rural accommodations, small villages, or countryside areas, taken in close-up, medium-distance, or wide shots.
        - Urban: Images depicting urban elements, such as houses, streets, or parks, taken in close-up or wide shots.
        - Sun & beach: Images depicting sun and beach tourism, such as people or events on the beach in a leisure context.
        - Other type: Images related to CES but not fitting any of the above categories.
        - Not Relevant: Images that are not relevant for Cultural Ecosystem Services studies.
        
        Return only the exact category name from the list above. Do not add any explanations, descriptions or additional information.
        """
        
        self.prompt_3 = """
        You are an image classification system specialized in Cultural Ecosystem Services (CES). Your task is to classify the given image into exactly one of the predefined categories below. Return only the category name and nothing else. Do not provide explanations or additional text.

        Categories:
        Landforms: Images primarily depicting land formations found in various landscapes, such as mountainous landscapes with high-altitude lagoons, coastal landscapes with visible bays and beaches, or river courses with riparian vegetation.
        Other abiotic features: Images primarily depicting other non-living elements not covered in the "Landforms" category, such as the sky and the sea. Also includes close-up images of snow (e.g., snowflakes, snowballs) that cannot be located in mountainous landscapes.
        Vegetation and habitats: Images primarily depicting vegetation and specific habitats (e.g., forests, farmland).
        Animals: Images primarily depicting animals or parts of them.
        Fungus: Images primarily depicting fungi or parts of them.
        Plants: Images primarily depicting plants or parts of them.
        Bridge: Images primarily depicting bridges.
        Roads: Images primarily depicting roads.
        Tracks and trails: Images primarily depicting paths, trails, and forest tracks.
        Vehicle: Images primarily depicting vehicles of any kind (bus, car, train, motorcycle, etc.).
        Accommodation: Images primarily depicting accommodations or their elements (e.g., house, bedroom, courtyard, etc.).
        Commerce facilities: Images primarily depicting commercial facilities (e.g., local shops, markets, supermarkets, restaurants, etc.).
        Gardens: Images primarily depicting gardens.
        Shelter: Images primarily depicting shelters (e.g., mountain refuge, cabin, etc.).
        Towns and villages: Images primarily depicting towns or their elements (e.g., squares, streets, etc.).
        Cities: Images primarily depicting cities or their elements (e.g., squares, streets, etc.).
        Dam: Images primarily depicting dams.
        Wind farm: Images primarily depicting wind farms.
        Breakwater: Images primarily depicting breakwaters.
        Dock: Images primarily depicting docks.
        Lighthouse: Images primarily depicting lighthouses.
        Heritage and culture: Images primarily depicting elements related to cultural heritage.
        Knowledge: Images primarily depicting scientific knowledge generation or environmental education activities.
        Spiritual, symbolic and related connotations: Images primarily depicting spiritual or religious experiences.
        Air activities: Images primarily depicting the practice of air sports.
        Terrestrial activities: Images primarily depicting the practice of terrestrial sports and other recreational activities in nature.
        Water activities: Images primarily depicting the practice of water sports and other recreational activities in water or on the beach.
        Winter activities: Images primarily depicting the practice of winter sports or other recreational activities in the snow.

        Return only the exact category name from the list above. Do not add any explanations or additional information.
        """

        self.prompt_4 = """
        You are an image classification system specialized in Cultural Ecosystem Services (CES). Your task is to classify the given image into exactly one of the predefined categories below. Return only the category name and nothing else. Do not provide explanations or additional text.

        Categories:
        Landforms: Images primarily depicting land formations found in various landscapes, such as mountainous landscapes with high-altitude lagoons, coastal landscapes with visible bays and beaches, or river courses with riparian vegetation.
        Other abiotic features: Images primarily depicting other non-living elements not covered in the "Landforms" category, such as the sky and the sea. Also includes close-up images of snow (e.g., snowflakes, snowballs) that cannot be located in mountainous landscapes.
        Vegetation and habitats: Images primarily depicting vegetation and specific habitats (e.g., forests, farmland).
        Animals: Images primarily depicting animals or parts of them.
        Fungus: Images primarily depicting fungi or parts of them.
        Plants: Images primarily depicting plants or parts of them.
        Bridge: Images primarily depicting bridges.
        Roads: Images primarily depicting roads.
        Tracks and trails: Images primarily depicting paths, trails, and forest tracks.
        Vehicle: Images primarily depicting vehicles of any kind (bus, car, train, motorcycle, etc.).
        Accommodation: Images primarily depicting accommodations or their elements (e.g., house, bedroom, courtyard, etc.).
        Commerce facilities: Images primarily depicting commercial facilities (e.g., local shops, markets, supermarkets, restaurants, etc.).
        Gardens: Images primarily depicting gardens.
        Shelter: Images primarily depicting shelters (e.g., mountain refuge, cabin, etc.).
        Towns and villages: Images primarily depicting towns or their elements (e.g., squares, streets, etc.).
        Cities: Images primarily depicting cities or their elements (e.g., squares, streets, etc.).
        Dam: Images primarily depicting dams.
        Wind farm: Images primarily depicting wind farms.
        Breakwater: Images primarily depicting breakwaters.
        Dock: Images primarily depicting docks.
        Lighthouse: Images primarily depicting lighthouses.
        Heritage and culture: Images primarily depicting elements related to cultural heritage.
        Knowledge: Images primarily depicting scientific knowledge generation or environmental education activities.
        Spiritual, symbolic and related connotations: Images primarily depicting spiritual or religious experiences.
        Air activities: Images primarily depicting the practice of air sports.
        Terrestrial activities: Images primarily depicting the practice of terrestrial sports and other recreational activities in nature.
        Water activities: Images primarily depicting the practice of water sports and other recreational activities in water or on the beach.
        Winter activities: Images primarily depicting the practice of winter sports or other recreational activities in the snow.
        Not Relevant: Images that are not relevant for Cultural Ecosystem Services studies.

        Return only the exact category name from the list above. Do not add any explanations or additional information.
        """

        
        # self.prompt_5 =  """
        # You are an image classification system specialized in Cultural Ecosystem Services (CES). Your task is to classify the given image into exactly one of the predefined categories below. Return only the category name and nothing else. Do not provide explanations or additional text.

        # Categories:
        # Nature & Landscape/Seascape: Images primarily depicting nature or landscapes, taken in wide shots.
        # Fauna/Flora: Images primarily depicting animals, plants, or their elements, taken in close-up or medium-distance shots.
        # Recreational: Images showing people in public recreational spaces engaging in casual, leisure-based activities with no competitive or structured sports context. Examples include families having a picnic or camping, people walking in a park, children playing informally, or people resting in nature. Food and drink may be present, but they should not be the central focus of the image.
        # Sports: Images focused on active sports participation, where the main theme is physical activity with a structured or competitive aspect. The image should clearly depict individuals practicing a sport with specific sports-related elements, such as courts, fields, tracks, goalposts, nets, or appropriate sports attire and equipment (e.g., soccer ball, tennis racket, bicycles, surfboards, ski poles). It may include organized events, training sessions, or competitive matches.
        # Cultural: Images depicting elements of cultural heritage, traditions, arts, and local craftsmanship. Examples include traditional weaving, pottery, folkloric dances, historical buildings with cultural significance, or traditional markets. If a religious site (e.g., a church) is shown as an architectural or historical landmark without active worship, it should be classified as Cultural.
        # Religious: Images emphasizing religious or spiritual elements, such as images of the Virgin Mary, processions, churches, pilgrimages, people in prayer, or places of worship actively being used for religious ceremonies. If the focus is on faith, devotion, or religious practice, classify it as Religious.
        # Gastronomy: Images where food, beverages, or culinary experiences are the primary focus. This includes dishes, meals at restaurants, traditional food markets, cooking processes, and food preparation. If people are present, the emphasis should be on the act of eating, drinking, or cooking rather than general leisure activities.
        # Rural tourism: Images depicting rural tourism elements, such as rural accommodations, small villages, or countryside areas, taken in close-up, medium-distance, or wide shots.
        # Urban: Images depicting urban elements, such as houses, streets, or parks, taken in close-up or wide shots.
        # Other type: Images related to CES but not fitting any of the above categories.
        # Sun & Beach: Images depicting tourism, leisure, or recreational activities occurring in a beach environment. The presence of sand, sea, or direct sun exposure in a coastal setting should result in this classification. People may be engaged in leisure, relaxation, sports, or events, but if the beach, sand and sun are present, classify it as Sun & Beach.
        # Not Relevant: Images that are not relevant for Cultural Ecosystem Services studies.

        # Return only the exact category name from the list above. Do not add any explanations or additional information.
        # """

        self.prompt_5 =  """
        You are an image classification system specialized in Cultural Ecosystem Services (CES). Your task is to classify the given image into exactly one of the predefined categories below. Return only the category name and nothing else. Do not provide explanations or additional text.

        Categories:
        Class_1: Images primarily depicting nature or landscapes, taken in wide shots.
        Class_2: Images primarily depicting animals, plants, or their elements, taken in close-up or medium-distance shots.
        Class_3: Images showing people in public recreational spaces engaging in casual, leisure-based activities with no competitive or structured sports context. Examples include families having a picnic or camping, people walking in a park, children playing informally, or people resting in nature. Food and drink may be present, but they should not be the central focus of the image.
        Class_4: Images focused on active sports participation, where the main theme is physical activity with a structured or competitive aspect. The image should clearly depict individuals practicing a sport with specific sports-related elements, such as courts, fields, tracks, goalposts, nets, or appropriate sports attire and equipment (e.g., soccer ball, tennis racket, bicycles, surfboards, ski poles). It may include organized events, training sessions, or competitive matches.
        Class_5: Images depicting elements of cultural heritage, traditions, arts, and local craftsmanship. Examples include traditional weaving, pottery, folkloric dances, historical buildings with cultural significance, or traditional markets. If a religious site (e.g., a church) is shown as an architectural or historical landmark without active worship, it should be classified as Cultural.
        Class_6: Images emphasizing religious or spiritual elements, such as images of the Virgin Mary, processions, churches, pilgrimages, people in prayer, or places of worship actively being used for religious ceremonies. If the focus is on faith, devotion, or religious practice, classify it as Religious.
        Class_7: Images where food, beverages, or culinary experiences are the primary focus. This includes dishes, meals at restaurants, traditional food markets, cooking processes, and food preparation. If people are present, the emphasis should be on the act of eating, drinking, or cooking rather than general leisure activities.
        Class_8: Images depicting rural tourism elements, such as rural accommodations, small villages, or countryside areas, taken in close-up, medium-distance, or wide shots.
        Class_9: Images depicting urban elements, such as houses, streets, or parks, taken in close-up or wide shots.
        Class_10: Images related to CES but not fitting any of the above categories.
        Class_11: Images depicting tourism, leisure, or recreational activities occurring in a beach environment. The presence of sand, sea, or direct sun exposure in a coastal setting should result in this classification. People may be engaged in leisure, relaxation, sports, or events, but if the beach, sand and sun are present, classify it as Sun & Beach.
        Class_12: Images that are not relevant for Cultural Ecosystem Services studies.

        Return only the exact category name from the list above. Do not add any explanations or additional information.
        """



        if n_prompt == 1:
            self.prompt = self.prompt_1 
        elif n_prompt == 2:
            self.prompt = self.prompt_2
        elif n_prompt == 3:
            self.prompt = self.prompt_3
        elif n_prompt == 4:
            self.prompt = self.prompt_4   
        elif n_prompt == 5:
            self.prompt = self.prompt_5       
        else:
            self.prompt = self.prompt_2




    def show_prompts(self):
        pass


    def run(self):
        self.__run_llava() if self.model == "llava1-5_7b" else self.__run_llava_next()

    
    def __run_llava(self):
        """
        Run Llava inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_csv) and self.cache:
            print("Recovering results from cache")
            self.result_df = pd.read_csv(self.results_csv, sep=";", header=0) 
        else:
            processor = LlavaProcessor.from_pretrained(self.model_hf)
            model = LlavaForConditionalGeneration.from_pretrained(self.model_hf, 
                                                                  torch_dtype=torch.float16, 
                                                                  low_cpu_mem_usage=True)
            model.to("cuda:0")

            results = []
            print(f"Launching llava: {self.model_hf}")
            
            for image_path in self.images:
                image = Image.open(image_path)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image", "image": image},  
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
                
                start_time = time.time()
                output = model.generate(**inputs, max_new_tokens=500)
                classification_result = processor.decode(output[0], skip_special_tokens=True)
                classification_category = classification_result.split(":")[-1].strip()
                inference_time = time.time() - start_time

                results.append({
                    "img": image_path,
                    "category_llava": classification_category,
                    "output": classification_result,
                    "inference_time": inference_time
                })

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            self.result_df = results_df



    def __run_llava_next(self):
        """
        Run Llava-Next inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_csv) and self.cache:
            print("Recovering results from cache")
            self.result_df = pd.read_csv(self.results_csv, sep=";", header=0) 
        else:
            processor = LlavaNextProcessor.from_pretrained(self.model_hf)
            model = LlavaNextForConditionalGeneration.from_pretrained(self.model_hf, 
                                                                      torch_dtype=torch.float16, 
                                                                      low_cpu_mem_usage=True)
            model.to("cuda:0")
            model.config.pad_token_id = model.config.eos_token_id

            results = []
            print(f"Launching llava: {self.model_hf}")
            
            for image_path in self.images:
                try:
                    image = Image.open(image_path).convert("RGB")  # Ensure compatibility with the model
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image", "image": image},  
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

                    start_time = time.time()
                    output = model.generate(**inputs, max_new_tokens=500)

                    classification_result = processor.decode(output[0], skip_special_tokens=True)
                    
                    if "[/INST]" in classification_result:
                        classification_category = classification_result.split("[/INST]")[-1].strip()
                    else:
                        classification_category = "Unknown"  # Handle unexpected output format

                    inference_time = time.time() - start_time

                    results.append({
                        "img": image_path,
                        "category_llava": classification_category,
                        "output": classification_result,
                        "inference_time": inference_time
                    })
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            self.result_df = results_df



    # def get_categories(self):
    #     """
    #     Returns categories from classification_lvl
    #     """
    #     return [cat.upper() for cat in self.categories]



    def get_results(self):
        """
        Returns inference results for given model name 
        (on classification_lvl where it was created, and for given prompt)
        """
        results = None
        try:
            results = pd.read_csv(self.results_csv,
                                  sep=";",
                                  header=0)
            results['category_llava'] = results['category_llava'].apply(lambda x: x.upper())
        except:
            ValueError("File not found")

        return results





if __name__ == "__main__":

    # Load images
    data_path = "data/flickr/flickr_validated_imgs_7000"
    url = Path(__file__).resolve().parent / data_path
    bbdd = "flickr"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    # Find all image files recursively and filter by extension (lowercase only)
    image_paths = [img_path for img_path in url.rglob('*') if img_path.suffix.lower() in image_extensions]
    # Convert to lowercase and remove duplicates (especially relevant for Windows)
    unique_image_paths = {img_path.resolve().as_posix().lower(): img_path for img_path in image_paths}
    images =  list(unique_image_paths.values())

    images_test_path = "images_to_analyze.csv"
    url_csv = Path(__file__).resolve().parent / images_test_path
    img_to_analyze = pd.read_csv(url_csv,sep=";", header=0, index_col=0)
    filtered_images = [img for img in images if str(img.resolve()).split("/")[-1] in list(img_to_analyze["img"])]

    # # Execute llava inference
    llava = LlavaInference(filtered_images,bbdd,"test",5,"llava1-6_7b",False,False)
    llava.run()

 
