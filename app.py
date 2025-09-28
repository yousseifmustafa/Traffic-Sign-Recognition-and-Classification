# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. الإعدادات وتحميل الموديل ---

# استخدام الكاش لضمان تحميل الموديل مرة واحدة فقط بكفاءة
@st.cache_resource
def load_traffic_model():
    """
    يقوم بتحميل الموديل المدرب مسبقًا.
    """
    try:
        # تأكد من أن المسار صحيح بالنسبة لمكان تشغيل التطبيق
        model = tf.keras.models.load_model("./trained_model/best_model.h5")
        return model
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
        return None

# قاموس لأسماء الفئات (Classes)
CLASSES = {
    0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h', 3: 'Speed Limit 60 km/h',
    4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h', 6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h',
    8: 'Speed Limit 120 km/h', 9: 'No Overtaking', 10: 'No Overtaking for Trucks', 11: 'Crossroads Ahead',
    12: 'Priority Road', 13: 'Give Way', 14: 'Stop', 15: 'No Vehicles',
    16: 'Vehicles over 3.5 tons prohibited', 17: 'No Entry', 18: 'General Caution',
    19: 'Dangerous Curve to the Left', 20: 'Dangerous Curve to the Right', 21: 'Double Bend First to the Left',
    22: 'Uneven Road', 23: 'Slippery Road', 24: 'Road Narrows on the Right', 25: 'Road Work',
    26: 'Traffic Signals', 27: 'Pedestrians Crossing', 28: 'Children Crossing', 29: 'Bicycles Crossing',
    30: 'Beware of Ice/Snow', 31: 'Wild Animals Crossing', 32: 'End of All Speed and Passing Limits',
    33: 'Turn Right Ahead', 34: 'Turn Left Ahead', 35: 'Ahead Only', 36: 'Go Straight or Right',
    37: 'Go Straight or Left', 38: 'Keep Right', 39: 'Keep Left', 40: 'Roundabout Mandatory',
    41: 'End of No Passing', 42: 'End of No Passing by Trucks',
}

# --- 2. دوال معالجة الصورة والتنبؤ ---

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    تقوم بمعالجة الصورة لتكون جاهزة للموديل.
    """
    img = image.resize((30, 30))
    img_array = np.array(img)
    # التأكد من أن الصورة لديها 3 قنوات لونية (RGB)
    if img_array.ndim == 2: # لو كانت الصورة رمادية
        img_array = np.stack((img_array,) * 3, axis=-1)
    if img_array.shape[-1] == 4: # لو كانت الصورة بها قناة شفافية (RGBA)
        img_array = img_array[:, :, :3]
    # إضافة بعد إضافي للـ batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model: tf.keras.Model, image: Image.Image) -> tuple[int, float]:
    """
    تقوم بعمل تنبؤ باستخدام الموديل المحمل.
    """
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return class_index, confidence

# --- 3. واجهة المستخدم (Streamlit UI) ---

# إعدادات الصفحة
st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦", layout="wide")

# تصميم الواجهة باستخدام CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4B4B4B; text-align: center; font-weight: bold; }
    .sub-header { text-align: center; color: #808080; }
    .result-box { background-color: #F0F2F6; padding: 25px; border-radius: 10px; text-align: center; }
    .result-text { font-size: 1.8rem; font-weight: bold; color: #1E88E5; }
    .confidence-text { font-size: 1.2rem; color: #555; }
</style>
""", unsafe_allow_html=True)

# عنوان التطبيق
st.markdown('<p class="main-header">Traffic Sign Classifier 🚦</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">تطبيق ذكاء اصطناعي للتعرف على إشارات المرور من الصور.</p>', unsafe_allow_html=True)
st.markdown("---")

# تحميل الموديل
model = load_traffic_model()

# التأكد من أن الموديل تم تحميله بنجاح
if model:
    # أداة رفع الملفات
    uploaded_file = st.file_uploader(
        "اختر صورة إشارة مرور...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("👋 مرحبًا! من فضلك ارفع صورة للبدء.")
    else:
        image = Image.open(uploaded_file)
        
        # تقسيم الشاشة لعمودين
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="الصورة التي تم رفعها", use_column_width=True)
        
        with col2:
            with st.spinner("جاري التحليل..."):
                # الحصول على التنبؤ
                class_index, confidence = predict(model, image)
                class_name = CLASSES.get(class_index, "إشارة غير معروفة")

                # عرض النتيجة في صندوق منسق
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<h3>النتيجة</h3>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-text">{class_name}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="confidence-text">نسبة الثقة: {confidence*100:.2f}%</p>', unsafe_allow_html=True)
                st.progress(confidence)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("لم يتم تحميل الموديل. تأكد من وجود ملف الموديل في المجلد الصحيح.")

# الجزء السفلي من الصفحة
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: grey;">'
    'Project Created By: Youssef Mustafa Hussein'
    '</div>',
    unsafe_allow_html=True
)
