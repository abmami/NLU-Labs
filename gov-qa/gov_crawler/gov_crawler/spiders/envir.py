import scrapy


class EnvirSpider(scrapy.Spider):
    name = "envir"
    allowed_domains = ["test.com"]
    start_urls = ["http://www.environnement.gov.tn/faq/"]

    def parse(self, response):
        qas = response.css('li[itemprop="hasPart"]')
        for qa in qas:
            data = {}
            data['question'] = qa.css('h3::text').get()
            answer = qa.css('.jpfaqAnswer :not(.jpfaqQuestionHelpfulText)::text').getall()
            data['answer'] = [t.strip() for t in answer if t.strip()!='Yes' and t.strip()!='No' ]
            yield data
