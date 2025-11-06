class Figure:
    def __init__(self, caption_obj, image_obj, caption_txt, file_name):
        self.file_name = file_name

        # image
        self.image_x_min = image_obj['x_min']
        self.image_x_max = image_obj['x_max']
        self.image_y_min = image_obj['y_min']
        self.image_y_max = image_obj['y_max']

        # caption
        self.caption_x_min = caption_obj['x_min']
        self.caption_x_max = caption_obj['x_max']
        self.caption_y_min = caption_obj['y_min']
        self.caption_y_max = caption_obj['y_max']
        self.caption_text = caption_txt
    
    def __str__(self):
        return (
            F"FILE NAME: {self.file_name}\n"
            "================IMAGE===================\n"
            f"x: {self.image_x_min}-{self.image_x_max}\n"
            f"y: {self.image_y_min}-{self.image_y_max}\n"
            "================CAPTION===================\n"
            f"x: {self.caption_x_min}-{self.caption_x_max}\n"
            f"y: {self.caption_y_min}-{self.caption_y_max}\n"
            f"text: {self.caption_text}\n"
        )