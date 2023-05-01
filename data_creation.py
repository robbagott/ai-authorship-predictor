import openai
import re
import pandas as pd
import json
from tqdm import tqdm
from time import sleep

openai.api_key = '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'

example_1={"id":"4b5c1ead-bd68-41dd-8242-500a7d4e005a","content":"A tiny alien creature that washed up on the banks of a river in western Russia has locals and experts stumped. A tiny alien creature that washed up on the banks of a river in north western Russia has locals and experts stumped. At first glance it looks like something borne from the abdominal cavity of a Nostromo crew member to wreak space-horror havoc of Sigourney Weaver and the human race.\n\nBut the four-centimetre oddity that was found in the Leningrad region in the town of Sosnovy Bor by a woman named Tamara as she waded in the shallows of the Kovashi River, according to a local television news report.\n\nWith what appears to be an elongated skull, shrunken frame and taloned limb, Tamara's friends thought it was a mutant chicken embryo.\n\nBut Tamara was not convinced the creature she found -- and christened 'Kesha' -- was of such mundane origins.\n\nBiologist Yegor Zadareev at the Krasnoyarsk Institute of Biophysics agreed.\n\n\"It seems that this body is neither fish nor fowl \u2026 this creature has a mysterious skull, no neck and wings,\" he said according to a translation of an interview on Russian TV.\n\nKesha was to be sent to Moscow for further analysis, which is sufficiently vague to conjure images of top secret underground bunker laboratories, reverse-alien probes and mitochondrial sequencing.\n\nKesha had alien conspiracy theorists are dusting off their tin-foil hats.\n\nUFO Sighting Daily, whose other tops stories are \"UFO follows The Donald Trump Helicopter, Tells US Trump will be next President\", and \"City on Mars Inside Alien Skull Found In India\" is eagerly awaiting the results of further tests.\n\nBut Tamara's friends and their mutant chicken egg theory is closer to the money for Sosnovy Bor locals. The key word being 'mutant'.\n\nThe residents of the town, which is in the shadow of the Leningrad Nuclear Power Plant, are \"naturally suspicious\" of expert authority, reported the Australian duo behind the Mysterious Universe podcast report, tongues-firmly-in-cheeks.\n\nThe plant had a history of disastrous industrial accidents and cover-ups, according to a former Russian Federal Inspectorate for Nuclear and Radiation Safety, Vladimir Kuznetsov.\n\nThree people were killed when a cooling circuit unit ruptured the year the plant opened in 1975. Over last three decades there have been two fires, a radiation spill detected six kilometres from the sight and five other major accidents at the plant.\n\nIf it is a radioactive mutant spawned from a leak at the nuclear power that makes Kesha more Blinky the fish than a flesh-eating alien.\n\nIt's not the first time reports of a bite-sized extra terrestrial corpse has captivated the imagination of Earthlings.\n\nPerhaps Kesha is a distant relative of Ata, the alien-like oddity discovered in a ghost town of the Atacama Desert of Chile in 2003.\n\nThe bizarre 12-centimetre-long skeleton was the star of the UFO documentary Sirius, and paraded as persuasive evidence of alien life.\n\nThat was until an immunologist had the novel idea that a rigorous scientific analysis of the specimen might hold the answers.\n\nAfter mapping more than 500 million reads to a reference human genome and examining x-rays of the humanoid, immunologist and director of the National Heart, Lung, and Blood Institute's Proteomics Centre for Systems Immunology at Stanford Garry Nolan proved Ata was of this world.\n\nAta was human. Possibly a 22-week-old foetus with a severe form of a rare rapid ageing disease, progeria, and died in the womb or after premature birth.\n\nWill the same extensive analysis be carried out to determine the origins of tiny Kesha? Scientists say it could take years.\n\nWho knows? Perhaps Kesha could become the first confirmed proof of alien humanoid life by the time President Trump takes up his second term in the White House. The story first appeared on The Sydney Morning Herald.","title":"Bizarre alien corpse discovered in Russia has experts stumped","media-type":"News","source":"Coly Point Observer","published":"2015-09-03T11:52:07Z"}
example_2={"id":"fef007d8-b1d4-42b8-b9fe-086965297512","content":"WASHINGTON \u2014 Worldwide sales of semiconductors slipped 0.9% year-over-year to $27.9 billion in July on a 90-day moving average. \n \nGlobal sales were 0.4% lower than the revised June total of $28 billion, the Semiconductor Industry Association said. \n \nRegionally, sales in the Americas were roughly flat in July compared to last year, while sales in China increased nearly 6%. \n \n\u201cGlobal semiconductor sales have slowed somewhat this summer in part due to softening demand, normal market cyclicality, and currency devaluation in some regional markets,\u201d said John Neuffer, president and CEO, SIA. \u201cDespite these headwinds, year-to-date global sales through July are higher than at the same time last year, which was a record year for semiconductor revenues.\u201d \n \nRegionally, year-to-year sales increased 5.6% in China, 1% in Asia Pacific\/other, and 0.8% in the Americas. Sales were down 12.5% in Europe and 13.3% in Japan, in part due to currency devaluation. \n \nOn a month-to-month basis, sales increased 2.7% in Japan, 0.6% in China and 0.4% in Europe while falling 0.3% in the Americas and 2.5% in Asia Pacific\/other. \n \nSIA also announced a joint release with the Semiconductor Research Corp. of a report highlighting the urgent need for research investments to advance the burgeoning Internet of Things and develop other cutting-edge, semiconductor-driven innovations. \"Implementing the recommendations in the report will help the United States harness new technologies and remain the world\u2019s top innovator,\u201d Neuffer said. \n \nAll monthly sales numbers are compiled by the World Semiconductor Trade Statistics (WSTS) organization and represent a three-month moving average. \n \nRegister for PCB West, the Silicon Valley's largest printed circuit trade show, Sept. 15-17 at the Santa Clara Convention Center:\u00a0 pcbwest.com .","title":"Semi Sales Slip in July","media-type":"News","source":"Printed Circuit Design & Manufacture","published":"2015-09-08T18:48:04Z"}

def remove_formatting_codes(text):
  """Removes formatting codes from a string"""
  pattern = re.compile(r'[\n\u2028\u2029\u00A0\t]')
  return re.sub(pattern, ' ', text)

def count_words(text):
  """
  This function takes a string as input and returns the word count.
  """
  text = remove_formatting_codes(text)
  words = text.split()
  return len(words)

# def summarization(title, content, source):
#   def get_prompt(t, c, s):
#     prompt = f'I am an intelligent news summarization bot. I have been tasked with giving a descriptive one or two sentence summary of the following article titled "{t}" published by the news outlet "{s}": {remove_formatting_codes(c)}'
#     return prompt

#   gpt = GPT(engine="text-davinci-003",
#           temperature=0.5,
#           max_tokens=250)

#   set_openai_key(api_key)

#   prompt_1 = get_prompt(example_1["title"], example_1["content"], example_1["source"])
#   gpt.add_example(Example(prompt_1, "A small, unusual creature found on the banks of a river in western Russia has locals and experts stumped as to its origins, with some speculating it could be proof of alien life, while others believe it may be a radioactive mutant from the nearby Leningrad Nuclear Power Plant, and it will be sent to Moscow for further analysis."))

#   prompt_2 = get_prompt(example_2["title"], example_2["content"], example_2["source"])
#   gpt.add_example(Example(prompt_2, "Global semiconductor sales slipped 0.9% year-over-year to $27.9 billion in July due to softening demand, normal market cyclicality, and currency devaluation in some regional markets, according to the Semiconductor Industry Association, although year-to-date global sales through July are higher than at the same time last year."))

#   prompt = get_prompt(title, content, source)
#   response = gpt.get_top_reply(prompt)[len('output: '):].strip()
#   return response

# def write_article_with_summary(title, source, brief_summary, word_count):
#   def get_prompt(t, s, b, w):
#     prompt = f'I am an intelligent news writing bot capable of mimicking the style of real news writers. I have been tasked with writing an article with the title "{t}" that will be published in the news outlet "{s}". I will write the article to be {w} words long. The following is a brief summary of the article content: {b}'
#     return prompt

#   gpt = GPT(engine="text-davinci-003",
#           temperature=0.5,
#           max_tokens=word_count)

#   set_openai_key(api_key)

#   prompt = get_prompt(title, source, brief_summary, word_count)
#   response = gpt.get_top_reply(prompt)[len('output: '):].strip()
#   return response

def write_article(title, source, word_count):
    def get_prompt(t, s, w):
        prompt = f'You have been tasked with writing an article with the title "{t}" that will be published in the news outlet "{s}". You will write the article to be {w} words long. Here is your article:'
        return prompt
  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [
            {"role": "system", "content": "You are an intelligent news writing bot capable of mimicking the style of real news writers"},
            {"role": "user", "content": get_prompt(title, source, word_count)},
        ]
    )

    content = str(response["choices"][0]["message"]["content"])
    return content

def main():
    INPUT_JSON = 'data/test.jsonl' # 'data/signalmedia-1000.jsonl'
    OUTPUT_PATH = 'data/test.csv' # 'data/train.csv' 

    jsonObj = pd.read_json(path_or_buf=INPUT_JSON, lines=True)
    news_articles = jsonObj[jsonObj['media-type']=='News']

    data = {'fake': [], 'real': []}
    count = 0
    for i in tqdm(news_articles.index):
        news_article = news_articles.iloc[i]

        word_count = count_words(news_article["content"])

        try:
            fake_article = remove_formatting_codes(write_article(news_article["title"], news_article["source"], word_count))
        except:
            print("Exception occurred writing article, sleeping for 5 seconds and skipping it.")
            sleep(5)
            continue

        real_article = remove_formatting_codes(news_article["content"])

        data['fake'].append(fake_article)
        data['real'].append(real_article)

        count += 1
        if count % 30 == 0:
            df = pd.DataFrame.from_dict(data)
            df.to_csv(OUTPUT_PATH)

    df = pd.DataFrame.from_dict(data)
    data_cleaning(df, OUTPUT_PATH)
    # df.to_csv('data/chat_gpt_real_fake_data.csv')

def data_cleaning(df, output_path):
    # df = pd.read_csv('data/chat_gpt_real_fake_data.csv')

    data = {'fake': [], 'real': []}
    for i in tqdm(df.index):
        data['fake'].append(re.sub('\s{2,}', ' ', df.iloc[i]["fake"]))
        data['real'].append(re.sub('\s{2,}', ' ', df.iloc[i]["real"]))
    
    df_new = pd.DataFrame.from_dict(data)
    df_new.to_csv(output_path)

if __name__=="__main__":
   main()