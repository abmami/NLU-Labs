import scrapy


class SocialFaq(scrapy.Spider):
    name = "social"


    def start_requests(self):
        start_urls = [
        'https://www.social.gov.tn/fr/faq?keyword=&service=All&page=0',
        'https://www.social.gov.tn/fr/faq?keyword=&service=All&page=1'
        ]
        for url in start_urls:
            yield scrapy.Request(
                url = url,
                callback=self.parse
            ) 

    def parse(self, response):
        
        data={}
        qas = response.css('.accordion-item.views-row > .accordion-item.views-row ')

        for qa in qas:
            data['question'] = qa.css('a::text').get()
            answer_content = qa.css('.field-content')
            text = answer_content.css(':not(a)::text').getall()
            data['answer'] = [t.strip() for t in text if t.strip()]

            yield data

