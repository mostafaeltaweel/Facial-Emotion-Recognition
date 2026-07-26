# Facial Emotion Recognition — Streamlit App

مشروع تخرج: تصنيف المشاعر من الوجه (Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise)
باستخدام EfficientNet-B3 + CBAM، مع واجهة Streamlit وكاميرا مباشرة.

## بنية المشروع
```
emotion_app/
├── app.py              # واجهة Streamlit (كاميرا حية + رفع صورة)
├── model.py             # بنية الموديل (EfficientNet-B3 + CBAM) + تحميل الأوزان
├── preprocessing.py      # نفس المعالجة المستخدمة في التدريب بالضبط
├── requirements.txt
└── best_model_ferplus.pth   # ⚠️ ملف الأوزان — لازم تحطه أنت هنا (غير موجود حاليًا)
```

## خطوات التشغيل

### 1. ثبّت المتطلبات
```bash
pip install -r requirements.txt
```

### 2. حط ملف الموديل المدرّب
انزل ملف `best_model_ferplus.pth` (أو أي checkpoint دربته من النوتبوك) وحطه
في نفس مجلد `app.py`. لو اسم الملف مختلف، عدّل المتغير `MODEL_PATH` في أول `app.py`.

> **مهم:** استخدمت بنية الموديل + أسماء الطبقات (`ca`, `sa`) المطابقة تمامًا لـ
> Cell 22 / Cell 25 في النوتبوك بتاعك (نسخة تدريب FER+ و RAF-DB). لو الملف اللي
> عندك اتدرب بنسخة الموديل الأولى (Cell 4 / Cell 8، الأسماء `channel_att`/`spatial_att`)
> رح تحتاج تعدّل `model.py` بحيث تتطابق أسماء الطبقات مع الـ checkpoint، وإلا
> `load_state_dict` رح يرمي خطأ.

### 3. شغّل التطبيق
```bash
streamlit run app.py
```

## شنو سوينا بالضبط (حسب طلبك)

| المتطلب | التطبيق |
|---|---|
| اكتشاف الوجه | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| الكاميرا الحية | `streamlit-webrtc` (فيديو حقيقي، مو صورة واحدة) |
| الأداء | الموديل يشتغل كل 5 فريمات فقط (`DETECT_EVERY_N_FRAMES`)، والصندوق يبقى ظاهر بين الفريمات لتفادي الوميض |
| كروب الوجه | يقص المنطقة داخل bounding box قبل ما يدخل للموديل |
| التناسق مع التدريب | نفس preprocessing بالضبط: Bilateral Filter → CLAHE → Gamma Correction → Resize 224×224 → Normalize (ImageNet mean/std) |
| Confidence threshold | لو أعلى احتمال أقل من 40% يعرض "Uncertain" بدل ما يفرض تصنيف خاطئ (المتغير `CONFIDENCE_THRESHOLD`) |
| رفع صورة | تبويب إضافي بسيط بجانب الكاميرا الحية، بدون أي تعقيد إضافي بالموديل |

## تحسينات إضافية (سريعة وذات قيمة عالية)

| التحسين | وين مطبق | الفايدة |
|---|---|---|
| **Temperature Scaling** | كل التنبؤات (`TEMPERATURE` مقروءة من `temperature.json`) | يعاير ثقة الموديل عشان `CONFIDENCE_THRESHOLD = 40%` يعكس ثقة حقيقية، مش موديل متحمس زيادة عن اللزوم |
| **Test-Time Augmentation (TTA)** | تبويب رفع الصورة فقط (`use_tta=True`) | متوسط تنبؤ الصورة الأصلية + انعكاسها الأفقي → دقة أعلى شوي بدون أي تدريب إضافي. ما استخدمناها بالكاميرا الحية عشان ما تبطئ الأداء (استدلال مضاعف لكل فريم) |
| **Prediction Smoothing (`FaceTracker`)** | الكاميرا الحية بس | بيتتبع كل وجه بين الفريمات (بأقرب مركز صندوق) ويعرض متوسط آخر 6 قراءات (`SMOOTHING_WINDOW`) بدل قراءة فريم واحد لحاله — يمنع "قفز" التصنيف بين المشاعر بسرعة، وتجربة الكاميرا تبان أسلس وأثبت بكثير |

### خطوة إضافية مطلوبة منك
بعد ما يخلص التدريب بالنوتبوك، رح تلاقي ملف `/kaggle/working/temperature.json` — **حمّله وحطه بجانب `app.py`** مع ملف الموديل. لو مش موجود، التطبيق بيشتغل عادي بس بدون معايرة (T=1.0 افتراضيًا).

## ملاحظات مهمة قبل المناقشة (Defense)

1. **ترتيب الفئات (Emotion Labels)**: مبني على `sorted(os.listdir(train_path))`
   بالضبط متل ما سوى في التدريب: `['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']`.
   إذا الداتاسيت عندك رتب المجلدات بشكل مختلف، لازم تتأكد من `EMOTION_LABELS` في `model.py`.

2. **الأداء على CPU**: EfficientNet-B3 موديل نسبيًا كبير. على جهاز بدون GPU، الاستدلال
   كل 5 فريمات لازم يكون كافي لتجربة سلسة تقريبًا. إذا لسه بطيء، زيد رقم
   `DETECT_EVERY_N_FRAMES` (مثلاً 8 أو 10).

3. **Haar Cascade محدودية**: دقته تنخفض مع الزوايا الحادة أو الإضاءة الضعيفة جدًا —
   وهاد نقطة كويسة تذكرها بالمناقشة كـ "Future Work" (مثلاً الترقية لـ MediaPipe لاحقًا).

4. **الموديل ثابت (frozen) وقت الاستدلال**: `model.eval()` مفعّل، فـ BatchNorm/Dropout
   ما رح يتغيروا أثناء التشغيل الحي، تمامًا متل وقت التقييم بالنوتبوك.
