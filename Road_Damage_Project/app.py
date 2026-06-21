import os
import csv
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session

# --- [1] إعداد المنظومة والـ Logging الاحترافي المتكامل ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SmartCityLogger")

app = Flask(__name__)
app.secret_key = 'syrian_smart_city_ultimate_enterprise_2026'

# --- [2] إدارة وتأمين المسارات والمجلدات الحيوية ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # حد أقصى 16 ميجابايت

# قاعدة البيانات الموحدة للمستخدمين والمحافظ الرقمية
USERS_DB = {
    'admin': {'name': 'المسؤول المركزي', 'password': 'admin', 'balance': 1000000},
    'user': {'name': 'علي خليل', 'password': '123', 'balance': 75000}
}

# --- [3] تهيئة ومراقبة موديل YOLOv8 الميداني ---
try:
    from ultralytics import YOLO
    import cv2
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        logger.info("✅ تم تحميل موديل YOLOv8 بنجاح وجاهز للرصد الميداني الفوري.")
    else:
        model = None
        logger.warning(f"⚠️ تحذير: ملف الموديل best.pt غير موجود في المسار الرئيسي: {MODEL_PATH}")
except ImportError:
    model = None
    logger.error("❌ خطأ قاسي: مكتبة ultralytics أو cv2 غير مثبتة في بيئة عمل البايثون الحالية!")

# --- [4] محرك إدارة ملفات الـ CSV بأمان وبشكل مركزي ---
def safe_read_csv(filename):
    path = os.path.join(DATA_FOLDER, filename)
    rows = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        rows.append(row)
        except Exception as e:
            logger.error(f"خطأ أثناء قراءة ملف {filename}: {e}")
    return rows

def safe_write_csv(filename, row_data):
    path = os.path.join(DATA_FOLDER, filename)
    try:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row_data)
        logger.info(f"💾 تم أرشفة سطر بيانات جديد بنجاح في: {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ فشل كتابة البيانات في {filename}: {e}")
        return False

# --- [5] محرك المعالجة المطور والآمن (YOLOv8 Core Analysis Engine) ---
def process_yolo_image(image_path, location, current_time, filename, reporter_name, reporter_phone, user_damage_type, notes):
    """تحليل الصورة بفتحها عبر OpenCV مباشرة وضمان استخراج قيم الثقة الحقيقية والأرشفة فوراً"""
    detected_type = "سليم / لم يتم الرصد"
    priority = "عادية"
    confidence_rate = "0%"

    print(f"\n==================================================")
    print(f"🚀 بدء معالجة الصورة في الباك إند: {filename}")
    print(f"📂 المسار الكامل المستهدف: {image_path}")
    print(f"🔍 فحص وجود الملف على القرص: {os.path.exists(image_path)}")
    print(f"🧠 حالة موديل اليولو في الذاكرة: {model is not None}")

    if model and os.path.exists(image_path):
        try:
            img = cv2.imread(image_path)
            if img is not None:
                print(f"📸 أبعاد الصورة المفتوحة بنجاح: {img.shape}")
                results = model(img, conf=0.15)
                print(f"📦 عدد الصناديق (Boxes) المكتشفة بواسطة YOLO: {len(results[0].boxes)}")
                
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    class_id = int(box.cls[0])
                    raw_type = str(model.names[class_id]).lower()
                    conf_val = float(box.conf[0]) * 100
                    confidence_rate = f"{conf_val:.1f}%"
                    
                    print(f"🎯 رصد ناجح! الكلاس الخام: {raw_type} | نسبة الثقة الحقيقية: {confidence_rate}")
                    
                    translations = {
                        'pothole': 'حفرة طريق متضررة',
                        'crack': 'تصدعات وشروخ إسفلتية',
                        'potholes': 'حفرة طريق متضررة',
                        'cracks': 'تصدعات وشروخ إسفلتية',
                        'electric_pole': 'عطل عمود كهرباء',
                        '0': 'حفرة طريق متضررة',
                        '1': 'تصدعات إسفلتية'
                    }
                    detected_type = translations.get(raw_type, translations.get(str(class_id), f"ضرر مرصود ({raw_type})"))
                    
                    if any(x in raw_type for x in ['pole', 'electric', 'كهرباء']):
                        priority = "حرجة جداً"
                    elif any(x in raw_type for x in ['pothole', 'crack', 'حفرة', 'تصدع']):
                        priority = "متوسطة"
                    
                    annotated_frame = results[0].plot()
                    cv2.imwrite(image_path, annotated_frame)
                    print(f"🎨 تم رسم مربعات الرصد بنجاح وحفظ الصورة المعدلة.")
                else:
                    print("⚠️ الموديل لم يتعرف تلقائياً على أضرار بالصورة. سيتم التحويل للخيار الاحتياطي المختار يدويًا.")
                    if user_damage_type and user_damage_type != 'none':
                        backup_map = {
                            'pothole': 'حفرة طريق (بلاغ مواطن)',
                            'electric_pole': 'عطل عمود كهرباء (بلاغ مواطن)',
                            'crack': 'تشققات وشروخ (بلاغ مواطن)'
                        }
                        detected_type = backup_map.get(user_damage_type, "بلاغ عام")
                        confidence_rate = "تأكيد يدوي"
                        priority = "متوسطة"
            else:
                print("❌ فشل ذريع: مكتبة OpenCV لم تتمكن من قراءة ملف الصورة!")
                detected_type = "فشل في قراءة ملف الصورة"

        except Exception as e:
            print(f"❌ خطأ داخلي في محرك استدلال اليولو: {e}")
            detected_type = "خطأ في معالجة الموديل"
    else:
        print("❌ تعذر الفحص: إما أن ملف الموديل best.pt مفقود أو أن مسار الصورة غير صحيح!")

    print(f"📝 السطر النهائي الذي سيتم أرشفته بالـ CSV: {detected_type} | الثقة: {confidence_rate}")
    print(f"==================================================\n")

    safe_write_csv('reports.csv', [
        filename, detected_type, location, current_time, 
        priority, 'مكتمل', confidence_rate, reporter_name, reporter_phone, notes
    ])

# --- [6] قنوات ومستقبلات البيانات الحية (Routes) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    fullname = request.form.get('fullname')
    username = request.form.get('username', '').strip()
    password = request.form.get('password')
    initial_balance = int(request.form.get('initial_balance', 0))
    
    if username in USERS_DB:
        flash('اسم المستخدم مسجل مسبقاً في المنظومة الرقمية!', 'danger')
        return redirect(url_for('index'))
        
    USERS_DB[username] = {'name': fullname, 'password': password, 'balance': initial_balance}
    session['user_username'] = username
    session['user_name'] = fullname
    session['user_balance'] = initial_balance
    
    flash(f'تم إنشاء هويتك الرقمية بنجاح!', 'success')
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password')
        
        if not username and not password:
            return redirect(url_for('admin_dashboard'))
            
        if username in USERS_DB and USERS_DB[username]['password'] == password:
            session['user_username'] = username
            session['user_name'] = USERS_DB[username]['name']
            session['user_balance'] = USERS_DB[username]['balance']
            return redirect(url_for('index'))
        else:
            flash('بيانات الهوية الرقمية غير مطابقة لسجلات الباك إند!', 'danger')
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('تم تسجيل الخروج من المنظومة الرقمية.', 'info')
    return redirect(url_for('index'))

@app.route('/streetwatch', methods=['GET', 'POST'])
def streetwatch():
    if request.method == 'POST':
        reporter_name = request.form.get('reporter_name') or 'مواطن مجهول'
        reporter_phone = request.form.get('reporter_phone') or '-'
        damage_type = request.form.get('damage_type') or 'none'
        location = request.form.get('location') or 'غير محدد'
        notes = request.form.get('notes') or '-'
        
        file = request.files.get('image') or request.files.get('file')
        
        if file and file.filename != '':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"AI_{timestamp}_{file.filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            process_yolo_image(image_path, location, current_time, filename, reporter_name, reporter_phone, damage_type, notes)
            return render_template('success.html')
            
    return render_template('streetwatch.html')

@app.route('/billing', methods=['GET', 'POST'])
def billing_route():  
    if request.method == 'POST':
        method = request.form.get('method') or 'بطاقة إلكترونية'
        account = request.form.get('account') or 'غير محدد'
        amount = request.form.get('amount') or '0'
        bill_type = request.form.get('bill_type') or 'خدمات عامة'
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        safe_write_csv('payments.csv', [method, account, amount, bill_type, current_time])
        return redirect(url_for('admin_dashboard'))
    return render_template('billing.html')

@app.route('/emergency', methods=['GET', 'POST'])
def emergency_route():
    if request.method == 'POST':
        name = request.form.get('name') or 'مجهول الهوية'
        phone = request.form.get('phone') or 'بدون رقم'
        emg_type = request.form.get('type') or 'طوارئ عامة'
        address = request.form.get('address') or 'غير محدد'
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        notes = request.form.get('notes') or '-'
        
        safe_write_csv('emergency_reports.csv', [name, phone, emg_type, address, current_time, notes])
        return redirect(url_for('admin_dashboard'))
    return render_template('emergency.html')

# --- [7] لوحة الإدارة المركزية العليا المحدثة (Admin Dashboard) ---
@app.route('/admin')
def admin_dashboard():
    reports_list = safe_read_csv('reports.csv')
    payments_list = safe_read_csv('payments.csv')
    emergency_list = safe_read_csv('emergency_reports.csv')
    
    pothole_count = 0
    pole_count = 0
    clean_count = 0
    
    for row in reports_list:
        if len(row) > 1:
            val = row[1].lower()
            if 'حفرة' in val or 'pothole' in val or 'تصدع' in val or 'crack' in val:
                pothole_count += 1
            elif 'كهرباء' in val or 'pole' in val or 'أعمدة' in val:
                pole_count += 1
            else:
                clean_count += 1

    return render_template(
        'admin.html', 
        reports=reports_list[::-1], 
        payments=payments_list[::-1], 
        emergency=emergency_list[::-1],
        pothole_count=pothole_count,
        pole_count=pole_count,
        clean_count=clean_count
    )

if __name__ == '__main__':
    logger.info("🚀 جاري تفعيل سيرفر المدينة الذكية المطور والموحد على بورت 5000...")
    app.run(debug=True, port=5000)


import os
from flask import send_from_directory

@app.route('/custom_images/<filename>')
def custom_images(filename):
    # المسار الخاص بمجلد الصور على جهازك
    photo_dir = r"C:\Users\User\OneDrive\Desktop\Road_Damage_Project\photo"
    return send_from_directory(photo_dir, filename)