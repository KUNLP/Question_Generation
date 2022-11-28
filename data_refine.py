from tqdm import tqdm
# f_list = ['train', 'test']
#
# for f_name in f_list:
#     with open('./data/{}.txt'.format(f_name),'r',encoding='utf8') as infile, open('./data/baseline_{}with_ans_symbol_and_title.txt'.format(f_name),'w',encoding='utf8') as outfile:
#         data_dict = {}
#         for line in infile:
#             parsed_context, answer, question, orig_context = line.split('\t')
#             title, context = parsed_context.split(" [title] ")
#             title = ''.join(title.split()).replace("_", " ").strip()
#             # context = ''.join(context.split()).replace("_", " ").strip()
#             context = ''.join(context.split()).replace("_", " ").replace(title, "<sys>").strip()
#             answer = ''.join(answer.split()).replace("_", " ").strip()
#             # answer_start = context.index("[answer]")
#             context = context.replace("[answer]", "<mask>")
#             # question = ''.join(question.split()).replace("_", " ").strip()
#             question = ''.join(question.split()).replace("_", " ").replace(title, "<sys>").strip()
#             # print(title)
#             # print(context)
#             # print(answer)
#             # print(question)
#             outfile.write("\t".join([title, answer, context, question]) + '\n')
# #
# # from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
# # get_kobart_tokenizer(".")
# # get_pytorch_kobart_model(cachedir=".")

###########################################################################################################################
import json
import nltk

def sent_tokenizer(context):
    char_to_sent_id = []
    refine_context = context.split("\n")
    result_context = []
    for _, r_context in enumerate(refine_context):
        if not r_context:
            rr_context = [""]
        else:
            rr_context = nltk.sent_tokenize(r_context)
        sent_id = len(result_context)
        for offset, sub_seq in enumerate(rr_context):
            char_to_sent_id += [sent_id+offset]*(len(sub_seq)+1)
        result_context+=rr_context
        refine_context = " ".join(result_context)
    return refine_context, result_context, char_to_sent_id
    # a = len(context.replace("\n", " "))
    # b = len(' '.join(result_context))
    # for idx, r_context in enumerate(refine_context):
    #     if len(context[:len(" ".join(refine_context[:idx+1]))]) != len(" ".join(refine_context[:idx+1])):
    #         print("#####################################")
    #         print(idx)
    #         print(context[:len(" ".join(refine_context[:idx+1]))])
    #         print(" ".join(refine_context[:idx+1]))
    #
    # if len(context.replace("\n", " ")) != len(' '.join(result_context)):
    #     for e in range(len(context)):
    #         if context.replace("\n", " ")[e] != ' '.join(result_context)[e]:
    #             print(context.replace("\n", " ")[e-10:e+10])
    #             print(' '.join(result_context)[e-10:e+10])
    #             print(e)
    #     print("?")
def process(f_name, outfile):
    print('\n\n\n', f_name)

    num = 0
    nnum = 0
    a = 0
    b = 0
    # ofile = open('./ai_data/refine_{}.json'.format(f_name), 'w', encoding='utf8')
    with open('./ai_data/{}.json'.format(f_name), 'r', encoding='utf8') as infile:
        data_dict = json.load(infile)
        result_dict = {"data":[]}
        for document in tqdm(data_dict["data"]):
            title = document["title"]
            document_dict = {"title":title, "paragraphs":[]}
            for paragraph in document["paragraphs"]:

                context = " ".join([e for e in paragraph['context'].replace("\n", " ").split() if e])
                refine_context, split_context, char_to_sent_id = sent_tokenizer(context)
                paragraph_dict = {"context":refine_context, "split_context":split_context, "qas":[]}
                for qas in paragraph["qas"]:
                    id = qas['id']
                    question = qas['question']
                    level = qas['level']

                    answer = qas['answers'][0]
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    keyword_text = answer['keyword']
                    keyword_start = answer['keyword_start']
                    try:
                        nnum+=1
                        answerable_context = split_context[char_to_sent_id[answer_start]]
                        # evidence_context = split_context[char_to_sent_id[keyword_start]]
                        evidence_context = ' '.join(split_context[char_to_sent_id[max(0, answer_start)]-5: char_to_sent_id[answer_start]])
                    except:
                        num+=1
                        continue
                    if answer_text not in answerable_context:
                        continue
                    # if keyword_text not in evidence_context:
                    #     continue
                    if answer_text not in refine_context:
                        continue
                    #outfile.write("\t".join([title, answer, context, question]) + '\n')
                    outfile.write("\t".join([id, level, title, answer_text.replace("\n", " "), evidence_context.replace("\n", " "), answerable_context.replace(answer_text, "<mask>").replace("\n", " "), question.replace("\n", " ")]) + '\n')
        #             qas_dict = {'question':question, "id":id, "level":level, "answers":[{"text":answer_text, "answer_start":refine_context.index(answer_text), "keyword_text":keyword_text, "keyword_start":keyword_start}]}
        #
        #             paragraph_dict["qas"].append(qas_dict)
        #         document_dict["paragraphs"].append(paragraph_dict)
        #     result_dict["data"].append(document_dict)
        # json.dump(result_dict, ofile, indent='\t', ensure_ascii=False)
        #
        # for data in data_dict:
        #     q_id = data['question_id'].replace("\n", "")
        #     question = data["question"].replace("\n", "")
        #     if not data['evidence_sent']:
        #         continue
        #     evidence_sent = data['evidence_sent'][0].replace("\n", "")
        #
        #     answer = answer_dict[q_id][0].replace("\n", "")
        #     title = answer_dict[q_id][1]
        #     context = [e.replace("\n", "") for e in data['splited_sent'] if answer in e.replace("\n", "")]
        #
        #     if not context:
        #         continue
        #     context = " ".join(context).replace(answer, "<mask>")
        #     num +=1
        #     outfile.write("\t".join([title, answer, evidence_sent, context, question]) + '\n')

# def answer_dict_load(f_name):
#     result_dict = {}
#     with open('./ai_data/{}.json'.format(f_name),'r',encoding='utf8') as infile:
#         data_dict = json.load(infile)["data"]
#         for document in data_dict:
#             title = document['title']
#             for qas in document['qas']:
#                 q_id = qas['id']
#                 answer = qas['answer']['answer_text']
#                 result_dict[q_id] = [answer, title]
#
#     return result_dict
f_list = ["all_outdomain_aug"]
#
for f_name in f_list:
    outfile = open('./processed_ai_data/{}.txt'.format(f_name), 'w', encoding='utf8')
    process(f_name, outfile)
#
# from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
# get_kobart_tokenizer(".")
# get_pytorch_kobart_model(cachedir=".")