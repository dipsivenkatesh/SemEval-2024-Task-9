import json
s= """{
"option_index": 2,
"option_answer": "English teacher"
}

Explanation:
Sally took her exams in French to have fun, but none of her teachers could understand her work. However, the English teacher may have a good friend from France, which could explain why the teacher was able to grasp Sally's test.
"""
answer = s
print(answer)
answer = answer.split('}')
answer = answer[0].strip() + '}'
print(answer)
ans_map = json.loads(answer)
ind = ans_map.get("option_index")
print(ind)
