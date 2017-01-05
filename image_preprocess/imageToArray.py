import Image
import numpy as np

def main(argv):
    imArray= process_image(argv)

def process_image(argv):
    img = Image.open( argv)
    img.load()
    data = np.asarray( img, dtype="int32")
    
    
