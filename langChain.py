from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from helper import extractTextFromPdf, scrape_website

# 4. Build agent that makes sense of that data, CV, and job desc
llm = OpenAI(temperature=0.6)
prompt_name = PromptTemplate(
    input_variables=['CV', 'job_desc'],
    template="""
    This is my CV: {CV}, it has all of my experiences and skills,
    and this a scraped web page of a job I want to apply to {job_desc}.
    write me an email application to this job, based the information from my CV,
    and job description, a subject to the email, and the extract the email of the recipient.
    """
)
job = "https://www.jobbank.gc.ca/jobsearch/jobposting/39074081?source=searchresults"
pdfText = extractTextFromPdf('../../../Documents/IMpor/CVs/CV-ENG.pdf')
job_scrapped = scrape_website(job)

chain = LLMChain(llm=llm, prompt=prompt_name)
output = chain.run(CV= pdfText, job_desc=job_scrapped)
print(output)

