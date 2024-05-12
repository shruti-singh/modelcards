import json
from tqdm import tqdm
import pickle
import google.generativeai as genai
import pandas as pd
import os 

genai.configure(api_key="Enter your Gemini_API_KEY")

genai.GenerationConfig(
    candidate_count = None,
    stop_sequences= [],
    max_output_tokens = None,
    temperature = 0.1,
    top_p = 0.95,
    top_k = None
)


model = genai.GenerativeModel('models/gemini-pro')
model_name = 'models/gemini-pro'


papers = ['mBART', 'DistilBERT', 'Longformer', 'T5', 'SparseTransformer', 'ERNIE', 'PEGASUS', 'FNet', 'Reformer', 'BART', 'RoBERTa', 'TransformerXL', 'BigBird', 'MobileBERT', 'GPT2', 'ELECTRA', 'StructBert', 'MuRIL', 'SciBERT', 'Transformer', 'GPT', 'SpanBERT', 'ALBERT', 'XLNet', 'BERT-PLI']

for paper in tqdm(papers):
    if os.path.exists(f"modelcards/qa/longcontext/gemini_full_text/model_gen_ans/{paper}.xlsx"):
        print(paper, "Exists already")
        continue
    print(paper, "started")
    # ----------------Context - paper-full-text----------------
    with open(f"modelcards/qa/longcontext/gemini_full_text/longdocs/{paper.lower()}_1.json", "r") as file:
        data = json.load(file)

    full_text = ""
    for paragraph in data['paragraphs']:
        full_text += paragraph + "\n"
    #---------------------------------------------------------------


    df = pd.read_excel(f'modelcards/qa/longcontext/gemini_full_text/qa_sheets/QAData_{paper}.xlsx')

    # questions = (df['Paper Title'].to_list()[9:])        
    questions = (df[df.keys()[0]].to_list()[9:])
    ground_truth = (df['Unnamed: 5'].to_list()[9:])

    # ----------------------GT-------------------------------------

    # ----------------------------------------------------------------


    model_gen_ans = []
    for ques in tqdm(questions):
        ans_gen_prompt = "You are provided with a question about a language model research paper in NLP and DL. You are also provided with the paper above that contains the answer. Answer the question based on the provided text. Do not include any additional text in the output except the answer to the question.\n\nQuestion: " + ques + "\nAnswer:"
        ans_gen_prompt = "Paper: "+ full_text + "\n" + "\n" +ans_gen_prompt
        # print(ans_gen_prompt)
        # print(ques)

        try:
            completion = model.generate_content(ans_gen_prompt)
            answer_list = list(completion.text.split("\n"))
            gen_answer = ""
            for i in answer_list:
                gen_answer += i
        except Exception as e:
            print(e)
            gen_answer = "No answer generated"
        model_gen_ans.append(gen_answer)

    # create xlsx file for ques, GT answer, and model gen ans columns
    df = pd.DataFrame()
    df['question'] = questions
    df['GT'] = ground_truth
    df['model_gen_ans'] = model_gen_ans
    df.to_excel(f"modelcards/qa/longcontext/gemini_full_text/model_gen_ans/{paper}.xlsx", index=False)
