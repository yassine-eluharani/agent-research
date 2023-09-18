from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from helper import scrape_website

with open('./data/cv.txt', 'r') as file:
    cv_text = file.read()

#llm = OpenAI(model="text-davinci-003",verbose=True)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
#job = "https://www.jobbank.gc.ca/jobsearch/jobposting/39074081?source=searchresults"
#job_scrapped = scrape_website(job)
job_scrapped = ("""
graphics programmer - Edmonton, AB - Job posting - Job Bank Skip to job search Skip to main content Skip to "About this Web application" Switch to basic HTML version Language selection Français fr Government of Canada / Gouvernement du Canada Search Search website Search Job BankJob Bank Account menu Sign in Job seekers Employers Menu and search MenuMenu Account menu Sign in Job seekers Employers Main navigation menu Job search Career planning Labour market information Hiring Help About You are here: Job BankSearch Discover jobs Job Search jobsearch Job search tools Dashboard Search Alerts Match My jobs My resumes Job Search Search 135,116 job postings in Canada What: Where: Type to get suggestions All of Canada Current location Advanced Browse Search 2123216 Loading, please wait... Cancel graphics programmer Verified Posted on September 10, 2023 by Employer details Canada Steel Works Ltd. Job details Education: Bachelor's degree. or equivalent experience. Work setting: Relocation costs not covered by employer. Willing to relocate. Tasks: Write, modify, integrate and test software code. Maintain existing computer programs by making modifications as required. Identify and communicate technical problems, processes and solutions. Prepare reports, manuals and other documentation on the status, operation and maintenance of software. Assist in the collection and documentation of user's requirements. Assist in the development of logical and physical specifications. Research and evaluate a variety of software products. Program animation software to predefined specifications for interactive CDs, DVDs, video game cartridges and Internet-based applications. Program special effects software for film and video applications. Write, modify, integrate and test software code for e-commerce and other Internet applications. Consult with clients after sale to provide ongoing support. Computer and technology knowledge: ASP.NET. Basic. C. C++. Java. SQL. Security and safety: Criminal record check. Work conditions and physical capabilities: Fast-paced environment. Tight deadlines. Sitting. Personal suitability: Accurate. Client focus. Efficient interpersonal skills. Excellent oral communication. Excellent written communication. Initiative. Judgement. Organized. Team player. Experience: 1 year to less than 2 years. LocationEdmonton, AB Salary$39.83HOUR hourly / 37.5 hours per week Terms of employment Permanent employmentFull time Day, Morning Start date Starts as soon as possible vacancies 2 vacancies Verified Source Job Bank #2592257 Various locationsEdmonton, AB Overview Languages English Education Bachelor's degree or equivalent experience Experience 1 year to less than 2 years Work setting Relocation costs not covered by employer Willing to relocate Responsibilities Tasks Write, modify, integrate and test software code Maintain existing computer programs by making modifications as required Identify and communicate technical problems, processes and solutions Prepare reports, manuals and other documentation on the status, operation and maintenance of software Assist in the collection and documentation of user's requirements Assist in the development of logical and physical specifications Research and evaluate a variety of software products Program animation software to predefined specifications for interactive CDs, DVDs, video game cartridges and Internet-based applications Program special effects software for film and video applications Write, modify, integrate and test software code for e-commerce and other Internet applications Consult with clients after sale to provide ongoing support Experience and specialization Computer and technology knowledge ASP.NET Basic C C++ Java SQL Additional information Security and safety Criminal record check Work conditions and physical capabilities Fast-paced environment Tight deadlines Sitting Personal suitability Accurate Client focus Efficient interpersonal skills Excellent oral communication Excellent written communication Initiative Judgement Organized Team player Employment groups Help - Employment groups This employer promotes equal employment opportunities for all job applicants, including those self-identifying as a member of these groups: Indigenous people, Persons with disabilities, Newcomers to Canada, Older workers, Veterans, Visible minorities, Youth Employment groups - Help Membership in a group is not a job requirement. All interested applicants are strongly encouraged to apply.This employer is committed to providing all job applicants with equal employment opportunities, and promoting inclusion. If you self-identify as a member of any employment group, you are encouraged to indicate it in your application. Close Who can apply to this job? The employer accepts applications from:Canadian citizens and permanent or temporary residents of Canada.Other candidates with or without a valid Canadian work permit. Show how to apply Advertised until 2023-10-10 Important notice: This job posting was posted directly by the employer on Job Bank. The Government of Canada has taken steps to make sure it is accurate and reliable but cannot guarantee its authenticity. Report a problem with this job posting Any fields marked with an asterisk (*) are required. *What’s wrong? This job posting contains incorrect information *Inaccurate salary *Inaccurate job title *Email *Provide more details: Report potential misuse of Job Bank Thank you for your help! You will not receive a reply. For enquiries, please contact us. Job Match Cut down on your job search time by allowing employers to find you! Sign up now! Other job posting options Favourite Favourites Email Print Share this pageBloggerDiigoEmailFacebookGmailLinkedIn®MySpacePinterestredditTinyURLtumblrTwitterWhatsappYahoo! MailNo endorsement of any products or services is expressed or implied.Share this page Job market information graphics programmer NOC 21232 Edmonton Region Median wage 39.83 $/hour Explore this career Similar job postings ...within Edmonton (AB) software development programmer JXE AUTOMATION TECHNOLOGY CO., software developer Acode software developer IQ Interactive Inc. application programmer San Pro Technology Ltd Similar job postings Report a problem or mistake on this page Share this pageBloggerDiigoEmailFacebookGmailLinkedIn®MySpacePinterestredditTinyURLtumblrTwitterWhatsappYahoo! MailNo endorsement of any products or services is expressed or implied.Share this page Date modified: 2023-06-17 Related links Job Bank Support About us Introduction to Job Bank Our network Terms of use - Job seekers Terms of use - Employers About this Web application Contact information Terms and conditions Privacy Top of Page
""")

job_template = """
    {job_desc}, this is a web scrapped page of a job offer,
    Extract the information nessecary to write a job application email.
    And this is my CV: {CV_file},
    based on my CV write a job application email.
    In the email include these point:
        1/ I am from morocco.
        2/ I am currently working on my express entry visa for Cananda,
        3/ I would need a sponsership
"""

gptPrompt = """
this is my cv "{CV_file}",
write an email job application for this job description "{job_desc}" 
"""

job_prompt_template = PromptTemplate(
    input_variables=["job_desc", "CV_file"], 
    template=gptPrompt
)
chain = LLMChain(llm=llm, prompt=job_prompt_template)
output = chain.run({"job_desc":job_scrapped, "CV_file": cv_text})
print(output)

