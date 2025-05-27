# -*- coding: utf-8 -*-


#Load pre-train model for Arabic Named Entity Recognition
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
model = AutoModelForTokenClassification.from_pretrained("hatmimoha/arabic-ner")
tokenizer = AutoTokenizer.from_pretrained("hatmimoha/arabic-ner")

# Test with data sample


sequence ="© Reuters. الأسهم الأوروبية تتقدم في تداولات حذرة قبيل إعلان قرار المركزي الأوروبي Investing.com – إرتفعت مؤشرات البورصات الاوروبية الرئيسية بحذر خلال جلسة تداول اليوم الخميس، وذلك مع ترقب المستثمرين والمتداولين لقرار السياسة النقدية المقرر أن يعلنه البنك المركزي الأوروبي في ختام إجتماعه الذي يجري اليوم، على أن يعقبه كالعادة المؤتمر الصحفي لرئيس البنك السيد (ماريو دراجي). فخلال تداولات الفترة الصباحية لليوم، إرتفع كل من مؤشر يورو ستوكس 50 بنسبة 0.16٪، ومؤشر كاك 40 الفرنسي بنسبة 0.05٪، ومؤشر داكس 30 الألماني بنسبة 0.22٪. هذا وبقي المستثمرون حذرين بعد أن قام صندوق النقد الدولي بتخفيض توقعاته للنمو الاقتصادي العالمي لكامل عام 2016 إلى 3.1٪، من التوقعات السابقة والبالغة 3.2٪، ولكنه أبقى على توقعاته لعام 2017 كما هي عند نسبة نمو تبلغ 3.4٪. ويترقب المشاركون في الأسواق على إختلافها القرار الذي سيتخذه البنك المركزي الأوروبي في ختام إجتماعه المقرر اليوم الخميس لمعرفة ما إذا كان أصحاب القرار سيقومون بإتخاذ المزيد من تدابير تسهيل السياسة النقدية لتعويض تداعيات الخروج البريطاني من الإتحاد الأوروبي. هذا وتباين أداء أسهم القطاع المالي الرئيسية. فلقد إرتفعت أسهم البنوك الفرنسية مع تقدم أسعار أسهم سوسيتيه جنيرال (بورصة باريس:SOGN) بنسبة 0.12٪، وأسهم مواطنه بي ان بي باريبا (بورصة باريس:BNPP) بنسبة 0.15٪، بينما إنخفضت أسعار أسهم البنوك الألمانية ممثلة بأسهم كومرتس بانك (بورصة فرانكفورت:CBKG) التي تراجعت بنسبة 0.22٪ وأسهم مواطنه دويتشة بانك (بورصة فرانكفورت:DBKGn) بنسبة 0.29٪. كما سلكت البنوك في الدول الطرفية ذات الطريق، مع إرتفاع أسعار أسهم البنوك الإيطالية انتيسا سان باولو (بورصة ميلان:ISP) بنسبة 1.09٪، وأونيكرديت (بورصة ميلان:CRDI) بنسبة 1.42٪، في حين تراجعت أسهم البنوك الإسبانية، مع إنخفاض بي بي في اي (بورصة مدريد:BBVA) بنسبة 0.02٪، وبانكو سانتاندير (بورصة مدريد:SAN) بنسبة 0.18٪. من جهة أخرى، سقطت أسهم لوفتهانزا بنسبة ضخمة بلغت 7.93٪ بعد ان أعلنت شركة الخطوط الجوية الألمانية عن تخفيض توقعاتها للأرباح للعام الحالي، وقالت أن الحجوزات إلى أوروبا قد إنخفضت بشكل ملحوظ في ظل الهجمات الإرهابية التي تعرضت لها القارة مؤخراً والتي تسببت بحالة من عدم اليقين السياسي والإقتصادي كما ذكرت الشركة. وفي لندن إنخفض مؤشر فوتسي 100 بنسبة 0.8٪، تحت ضغط أسهم إيزي جيت التي سقطت بنسبة 6.12٪ بعد أن قالت الشركة أنه لا يمكن التكهن بنتائج العام الحالي من الإيرادات والأرباح وذلك بسبب المخاوف الأمنية المتزايدة وضعف ثقة المستهلك. وفي قطاع التعدين إرتفعت جميع الأسهم الرئيسية، وذلك مع تقدم أسعار أسهم كل من ريو تنتو بنسبة 0.92٪، وبي إتش بي بيلتون بنسبة 1.21٪، وإرتفاع أسهم كل من جلينكور بنسبة 1.93٪ وأنجلو اميريكان بنسبة 2.11٪. كما إرتفعت الأسهم الرئيسية في قطاع المال والبنوك، مع إرتفاع أسعار أسهم كل باركليز (بورصة لندن:BARC) بنسبة 0.83٪، ومجموعة إتش إس بي سي القابضة (بورصة لندن:HSBA) بنسبة 0.61٪، وأسهم مجموعة لويدز المصرفية (بورصة لندن:LLOY) بنسبة 0.09٪، ورويال بانك اوف سكوتلاند (بورصة لندن:RBS) بنسبة 0.10٪. وفي الولايات المتحدة، وقبل إفتتاح البورصات الأمريكية أبوابها لليوم، إرتفعت مؤشرات الأسهم الآجلة بنسب هامشية. فلقد تقدم كل من مؤشر داو جونز 30 للعقود الآجلة بنسبة 0.01٪، ومؤشر ستاندرد آند بورز 500 بنسبة 0.02٪، في حين اظهر ناسداك 100 إرتفاعاً أكثر حدة وقدره 0.09٪."


import re


def remove_punctuation(text):
    """Remove punctuation, English chars and numbers from text."""
    pattern = r"[^\w\s]|[\d]|[a-zA-Z]+"
    text = re.sub(pattern, "", text)
    return text

#To deal with long sequence
def split_text(text, max_length=512):
    """Split text into smaller chunks."""
    chunks = []
    start = 0
    text = remove_punctuation(text)
    while start < len(text):
        end = start + max_length
        if end >= len(text):
            chunks.append(text[start:])
            break
        else:
            end = text.rfind(" ", start, end)
            if end == -1:
                end = start + max_length
            chunks.append(text[start:end])
            start = end + 1
    return chunks


def get_ents(tokens, predictions):
 
  """
  Get only the 3 entities ORGANIZATION,LOCATION and PERSON,
  fix word tokens sub split like this ##ك to get readable words
  """
  org=[]
  loc=[]
  man=[]

  for token, prediction,index in zip(tokens, predictions[0].numpy(),list(range(len(tokens)))):
    if model.config.id2label[prediction].find('ORGANIZATION') != -1:
        if (token.find('##')!=-1) and len(org)>0and (org[-1][1] == index-1) :
            org[-1]= [org[-1][0]+token[2:],index]
        elif token.find('##')==-1:
            org.append([token,index])

    elif model.config.id2label[prediction].find('LOCATION') != -1:
        if (token.find('##')!=-1) and len(loc)>0 and (loc[-1][1] == index-1) :
            loc[-1]= [loc[-1][0]+token[2:],index]
        elif token.find('##')==-1:
            loc.append([token,index])

    elif model.config.id2label[prediction].find('PERSON') != -1:
        if (token.find('##')!=-1) and len(man)>0 and (man[-1][1] == index-1) :
            man[-1]= [man[-1][0]+token[2:],index]
        elif token.find('##')==-1:
            man.append([token,index])
  
  return [ i[0]for i in org],[ i[0]for i in loc],[ i[0]for i in man]

def get_pred(sequence):
  """The function call get_ents function and get tokens and predictions"""
  # Bit of a hack to get the tokens with the special tokens
  tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
  inputs = tokenizer.encode(sequence, return_tensors="pt")
  outputs = model(inputs).logits
  predictions = torch.argmax(outputs, dim=2)
  return get_ents(tokens, predictions)

def func(chunks):
  """This the main predict each chunk and combine the result and return dict"""

  s1 = set()
  s2 = set()
  s3 = set()

  for sequence in chunks:
     org,loc,man = get_pred(sequence)
     s1.update(org)
     s2.update(loc)
     s3.update(man)


  return {'Persons':list(s3),'Organizations':list(s1) , 'Locations':list(s2)}

func(split_text(sequence))
