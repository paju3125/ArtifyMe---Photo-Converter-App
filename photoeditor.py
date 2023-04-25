# Import libraries
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io

from PIL import Image, ImageDraw, ImageFont

def add_text(image, text, font_size, position):
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('arial.ttf', font_size)
    draw.text(position, text, font=font)
    return img

def add_sticker(image, sticker_path, position, resize=None):
    img = Image.open(image)
    sticker = Image.open(sticker_path)
    # Resize sticker if resize parameter is given
    if resize:
        sticker = sticker.resize((int(sticker.size[0]*resize), int(sticker.size[1]*resize)))
    img.paste(sticker, position, sticker)
    return img

# Save the processed image
def save_image(img_array):
    img = Image.fromarray(np.uint8(img_array))
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


def auto_enhance(image):
    import cv2

def auto_enhance(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Apply histogram equalization on the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    
    # Merge the equalized L channel with the original A and B channels
    lab_eq = cv2.merge((l_eq, a, b))

    # Convert the LAB image back to RGB color space
    enhanced_image = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load image
image = Image.open('logo13.png')

# Create two columns with different width
col1, col2 = st.columns([0.8, 0.2])
with col1:
    # Display header text using CSS style
    st.markdown("""
        <style>
            .font {
                font-size: 35px;
                font-family: 'Cooper Black';
                text-transform: uppercase;
	background: linear-gradient(to right, #e05858
 0%, #D98324 100%);
	-webkit-background-clip: text;
	-webkit-text-fill-color: transparent;
                
            }
            
            .more{
                font-size: 20px;
            }
            
            .title{
                font-size: 40px
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
with col2:
    # Display brand logo
    pass

# Add a header and expander in sidebar
st.sidebar.image(image, width=100, use_column_width=True)
# st.sidebar.markdown(
#     '<p class="font title">ArtifyMe - <br><span class="more">Photo Converter App</span></p>', unsafe_allow_html=True)
with st.sidebar.expander("About the app"):
    st.write("Edit your moments, enhance your memories!")

# Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Add 'before' and 'after' columns
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
        st.image(image, width=300)
    with col2:
        st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)
        filter = st.sidebar.radio('Edit your photo:', ['Original', 'Gray Image', 'Black and White', 'Pencil Sketch', 'Blur Effect', 'Rotate', 'Flip', 'Resize', 'Crop','Edge Detection', 'Brightness', 'Contrast', 'Saturation', 'Face Detection and Recognition', 'Text and Sticker Overlay', 'Automatic Photo Enhancement'])

        if filter == 'Automatic Photo Enhancement':
            # Apply automatic photo enhancement
            image = np.array(image)
            enhanced_image = auto_enhance(image)
            st.image(enhanced_image, width=300,clamp = True)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(enhanced_image)
                    st.download_button(label='Download', data=img_io, file_name='auto_enhanced_image.png', mime='image/png')
        
        # elif filter == 'Object Detection':
        #     image = np.array(image)
        #     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        #     img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            
        #     clock = cv2.CascadeClassifier('haarcascade_wallclock.xml')  
        #     found = clock.detectMultiScale(img_gray,  
        #                                 minSize =(20, 20)) 
        #     amount_found = len(found)
        #     st.text("Detecting a clock from an image")
        #     if amount_found != 0:  
        #         for (x, y, width, height) in found:
            
        #             cv2.rectangle(img_rgb, (x, y),  
        #                         (x + height, y + width),  
        #                         (0, 255, 0), 5) 
        #     st.image(img_rgb, width=300,clamp = True)

        elif filter == 'Face Detection and Recognition':
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            image = np.array(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            st.image(img, width=300)

        elif filter == 'Gray Image':
            converted_img = np.array(image.convert('RGB'))
            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
            st.image(gray_scale, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(gray_scale)
                    st.download_button(label='Download', data=img_io, file_name='gray.png', mime='image/png')
        
        elif filter == 'Black and White':
            converted_img = np.array(image.convert('RGB'))
            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
            slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
            (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
            st.image(blackAndWhiteImage, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(blackAndWhiteImage)
                    st.download_button(label='Download', data=img_io, file_name='black_and_white.png', mime='image/png')
        
        elif filter == 'Pencil Sketch':
            converted_img = np.array(image.convert('RGB'))
            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
            inv_gray = 255 - gray_scale
            slider = st.sidebar.slider('Adjust the intensity', 25, 255, 125, step=2)
            blur_image = cv2.GaussianBlur(inv_gray, (slider, slider), 0, 0)
            sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
            st.image(sketch, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(sketch)
                    st.download_button(label='Download', data=img_io, file_name='pencil_sketch.png', mime='image/png')
        
        elif filter == 'Blur Effect':
            converted_img = np.array(image.convert('RGB'))
            slider = st.sidebar.slider('Adjust the intensity', 5, 81, 33, step=2)
            converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(converted_img, (slider,slider), 0, 0)
            st.image(blur_image, channels='BGR', width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(blur_image)
                    st.download_button(label='Download', data=img_io, file_name='blured_image.png', mime='image/png')
        
        elif filter == 'Rotate':
            degrees = st.sidebar.slider('Rotate the image by how many degrees?', -180, 180, 0, 1)
            rotated_image = image.rotate(degrees)
            st.image(rotated_image, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(rotated_image)
                    st.download_button(label='Download', data=img_io, file_name='rotated_image.png', mime='image/png')
        
        elif filter == 'Flip':
            flip_option = st.sidebar.radio('Choose flip direction:', ['Horizontal', 'Vertical'])
            if flip_option == 'Horizontal':
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            st.image(flipped_image, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(flipped_image)
                    st.download_button(label='Download', data=img_io, file_name='flipped_image.png', mime='image/png')
        
        elif filter == 'Resize':
            width = st.sidebar.slider('New width:', 1, 1000, image.size[0], 1)
            height = st.sidebar.slider('New height:', 1, 1000, image.size[1], 1)
            resized_image = image.resize((width, height))
            st.image(resized_image, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(resized_image)
                    st.download_button(label='Download', data=img_io, file_name='resized_image.png', mime='image/png')
        
        elif filter == 'Crop':
            left = st.sidebar.slider('Left edge:', 0, image.size[0], 0, 1)
            top = st.sidebar.slider('Top edge:', 0, image.size[1], 0, 1)
            right = st.sidebar.slider('Right edge:', left, image.size[0], image.size[0], 1)
            bottom = st.sidebar.slider('Bottom edge:', top, image.size[1], image.size[1], 1)
            cropped_image = image.crop((left, top, right, bottom))
            st.image(cropped_image, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(cropped_image)
                    st.download_button(label='Download', data=img_io, file_name='cropped_image.png', mime='image/png')
        
        elif filter == 'Edge Detection':
            converted_img = np.array(image.convert('RGB')) 
            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
            low_threshold = st.sidebar.slider('Low threshold', 0, 255, 50)
            high_threshold = st.sidebar.slider('High threshold', 0, 255, 150)
            edges = cv2.Canny(gray_scale, low_threshold, high_threshold)
            st.image(edges, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(edges)
                    st.download_button(label='Download', data=img_io, file_name='edges.png', mime='image/png')
        
        elif filter == 'Brightness':
            converted_img = np.array(image.convert('RGB')) 
            brightness_factor = st.sidebar.slider('Brightness factor', 0.1, 3.0, 1.0, step=0.1)
            enhancer = ImageEnhance.Brightness(Image.fromarray(converted_img))
            brightened_img = np.array(enhancer.enhance(brightness_factor))
            st.image(brightened_img, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(brightened_img)
                    st.download_button(label='Download', data=img_io, file_name='brightened_img.png', mime='image/png')
        
        elif filter == 'Contrast':
            converted_img = np.array(image.convert('RGB')) 
            contrast_factor = st.sidebar.slider('Contrast factor', 0.1, 3.0, 1.0, step=0.1)
            enhancer = ImageEnhance.Contrast(Image.fromarray(converted_img))
            contrasted_img = np.array(enhancer.enhance(contrast_factor))
            st.image(contrasted_img, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(contrasted_img)
                    st.download_button(label='Download', data=img_io, file_name='contrasted_img.png', mime='image/png')
            
        elif filter == 'Saturation':
            converted_img = np.array(image.convert('RGB')) 
            saturation_factor = st.sidebar.slider('Saturation factor', 0.1, 3.0, 1.0, step=0.1)
            enhancer = ImageEnhance.Color(Image.fromarray(converted_img))
            saturated_img = np.array(enhancer.enhance(saturation_factor))
            st.image(saturated_img, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(saturated_img)
                    st.download_button(label='Download', data=img_io, file_name='saturated_img.png', mime='image/png')
            
        elif filter == 'Text and Sticker Overlay':
            # Create input fields for text and sticker
            text = st.text_input('Add text:')
            sticker_path = st.file_uploader('Add sticker:', type=['png', 'jpg', 'jpeg'])

            # Add both text and sticker to the image
            img = Image.open(uploaded_file)
            if text:
                # Create input fields for font size, position, and color
                text_position_x = st.slider('Text X:', 0, 500, 100)
                text_position_y = st.slider('Text Y:', 0, 500, 100)
                text_position = (text_position_x, text_position_y)
                font_size = st.slider('Font size:', 10, 100, 20)
                text_color = st.color_picker('Text color', '#000000')
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype('arial.ttf', font_size)
                draw.text(text_position, text, font=font, fill=text_color)
            if sticker_path:
                # Create input fields for font size and position
                sticker_position_x = st.slider('Sticker X:', 0, 500, 100)
                sticker_position_y = st.slider('Sticker Y:', 0, 500, 100)
                sticker_position = (sticker_position_x, sticker_position_y)
                # Create input fields for sticker resize
                sticker_resize = st.slider('Sticker resize:', 0.1, 2.0, 1.0, 0.1)
                sticker = Image.open(sticker_path)
                # Resize sticker if resize parameter is given
                if sticker_resize:
                    sticker = sticker.resize((int(sticker.size[0]*sticker_resize), int(sticker.size[1]*sticker_resize)))
                img.paste(sticker,sticker_position, sticker)
            st.image(img, width=300)
            # Add download button
            if st.button('Download Image'):
                with st.spinner('Downloading...'):
                    img_io = save_image(img)
                    st.download_button(label='Download', data=img_io, file_name='image_with_text_sticker.png', mime='image/png')

        else: 
            st.image(image, width=300) 


