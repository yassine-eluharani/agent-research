import streamlit as st
from helper import extractTextFromPdf, scrape_website

# jobbank url advanced search.
url = "https://www.jobbank.gc.ca/jobsearch/job_search_advanced.xhtml?fage=2&fn21=20012&fn21=21211&fn21=21222&fn21=21223&fn21=21230&fn21=21231&fn21=21232&fn21=21234&fn21=21311&fn21=22222&fper=F&fexp=2&fglo=1&sort=M&fsrc=16"
# Actual search results
urlResults = "https://www.jobbank.gc.ca/jobsearch/jobsearch?fage=2&fn21=20012&fn21=21211&fn21=21222&fn21=21223&fn21=21230&fn21=21231&fn21=21232&fn21=21234&fn21=21311&fn21=22222&fper=F&fexp=2&fglo=1&page=1&sort=M&fsrc=16"

# 5. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")
    st.header("AI research agent :bird:")
    query = st.text_input("url")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file and query:
        pdfText = extractTextFromPdf(uploaded_file)
        job_scrapped = scrape_website(query)
        result = prompt_name.format(CV= pdfText, job_desc=job_scrapped)
        st.write(result)

if __name__ == '__main__':
    main()
