import scrapy


class FinanceSpider(scrapy.Spider):
    name = "finance"
    def start_requests(self):
        start_urls = [f'http://www.finances.gov.tn/fr/faq?body_value=&field_themef_target_id=All&page={i}' for i in range(1,10)]
        for url in start_urls:
            yield scrapy.Request(
                url = url,
                callback=self.parse
            )


    def parse(self, response):
        qas = response.css('.faqRreponse')
        for qa in qas:
            data = {}
            data['question'] = qa.css('.question::text').get()
            data['answer'] = qa.css('.reponse::text').get()
            yield data