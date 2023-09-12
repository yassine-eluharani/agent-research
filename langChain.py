from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from helper import scrape_website

# 4. Build agent that makes sense of that data, CV, and job desc
CV_text="""
Education: Master's degree in Computer Engineering.
Work expreice 2 years:
Full Stack Developer: From January 2022 to January 2023 
- API and Microservices: Creating RESTful APIs with Python Django
within a microservices architecture for seamless interactions.
-DevOps: Leveraging Docker for efficient container management
and ensuring portability.
- Documentation: Thorough documentation of RESTful APIs using
OPEN API and SWAGGER to simplify integration.
- Dynamic Frontend: Designing new UI components using React.js
and Typescript for a modern and responsive user experience.
- Testing and Quality: Adopting a test-driven development
approach specifically focusing on unit testing React components
using Jest. Conducting end-to-end tests using Cypress to ensure
optimal quality.

Full Stack Developer Since May 2023
- Full Stack: Mastery of React.js for dynamic frontend and Tailwind
CSS for elegant interfaces. Using Node.js for full backend
development.
- API: Design and enhancement of RESTful APIs for seamless
interactions. Expertise in secure authentication using JSON Web
Tokens (JWT).
- Cloud: Practical experience with AWS services like EC2 for virtual
instances, S3 for storage, and CloudFront for content distribution.
- Payments: Expert integration of payment methods through
platforms such as Stripe and PayPal, ensuring safe and smooth
online transactions.
- DevOps: Creation and management of CI/CD pipelines using
Docker for efficient development and deployment processes.
Advanced utilization of GitHub Actions for continuous automation.

Skills: 
Programming Languages: Proficient in JAVA, Python, C, Typescript, and JavaScript. 
Web Technologies: Experienced with Node.js, Express.js, React, React-Native, NEXT.js, unit testing, RESTful APIs, microservices, Django, Spring Boot, GraphQL API, OPEN API, Pact, Cypress, Selenium.
Cloud Services: Familiar with S3 buckets, EC2 instances, API Gateway, CloudFront, SNS (Simple Notification Service), AWS Chime.
Environment and Tools: Skilled in Linux, Docker, Git version control, Shell scripting (BASH), Redpanda, Kafka, Agile methodology, Test- driven development (TDD), Continuous Integration/Continuous Deployment (CI/CD).
Databases: Proficient in Microsoft SQL Server, MySQL, PostgreSQL, Oracle PL/SQL, NoSQL databases, MongoDB, Firebase.
"""

llm = OpenAI(temperature=0.5)

job = "https://www.jobbank.gc.ca/jobsearch/jobposting/39074081?source=searchresults"
job_scrapped = scrape_website(job)

job_template = """
    {job_desc}, this is a web scrapped page of a job offer,
    Extract the information nessecary to write a job application email.
    And this is my CV: {CV_file},
    Also from my CV extract the useful nessecary informations to write a job application email.
"""
job_prompt_template = PromptTemplate(
    input_variables=["job_desc", "CV_file"], 
    template=job_template
)
job_chain = LLMChain(llm=llm, prompt=job_prompt_template, output_key="important_information")

email_template = """
Given these information {important_information} extracted from my cv and the job description,
Write a job application email, for that job using the extracted nessecary information from CV,
    In the email include these point:
    1/ I am from morocco.
    2/ I am currently working on my express entry visa for Cananda,
    3/ I would need a sponsership

"""
prompt_template = PromptTemplate(input_variables=["important_information"], template=email_template)
email_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="email")

overall_chain = SequentialChain(
    chains=[job_chain, email_chain],
    input_variables=["job_desc","CV_file"],
    output_variables=["email"],
    verbose=True)

output = overall_chain({"job_desc":job_scrapped, "CV_file": CV_text})
print(output)
