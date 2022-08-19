def extractiveSummarizationBert():
  # get_ipython().system('pip install PyPDF2')
  # get_ipython().system('pip install transformers')

  from PyPDF2 import PdfReader
  from transformers import pipeline
  from transformers import pipeline
  sentiment_pipeline = pipeline("sentiment-analysis")

  textForWeb = ''
  plots = []


  def remove_unnecessay_lines(text):

    final_text = ""

    for line in text.split('\n'):
      if len(line.split(' ')) > 1:
        final_text += line
    return final_text


  def extract_summary(pg_no, reader_obj, summary_obj):
    page = reader.pages[pg_no]
    text = page.extract_text()
    final_text = remove_unnecessay_lines(text)
    summary_ =  summarizer(final_text, max_length=500, min_length=30, do_sample=False)[0]['summary_text']
    return summary_


  summarizer = pipeline("summarization")


  reader = PdfReader("press-release-Q2FY22.pdf")

  summary = []

  for pgs in range(reader.getNumPages()):
    summary.append(extract_summary(pgs, reader, summarizer))



  print("Summary : \n\n")
  textForWeb = textForWeb + '<p align="left"> Summary : </p><br><p align="left">'
  for txt in summary:
    print(txt)
    textForWeb = textForWeb + txt + '</p><br><p align="left">'
  summary = ' '.join(summary)



  sentiment_pipeline([summary])
  textForWeb = textForWeb + str(sentiment_pipeline([summary])) + '</p><br><br><p align="left">'

  reader = PdfReader("q1fy22-press-release.pdf")

  summary = []

  for pgs in range(reader.getNumPages()):
    summary.append(extract_summary(pgs, reader, summarizer))

  print("Summary : \n\n")
  textForWeb = textForWeb + ' Summary : </p><br><p align="left">'
  for txt in summary:
    print(txt)
    textForWeb = textForWeb + txt + '</p><br><p align="left">'
  summary = ' '.join(summary)


  sentiment_pipeline([summary])
  textForWeb = textForWeb + str(sentiment_pipeline([summary])) + '</p>'

  return textForWeb, plots

# res = extractiveSummarizationBert()
# print(res)