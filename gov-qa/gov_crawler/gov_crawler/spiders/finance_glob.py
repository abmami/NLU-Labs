import scrapy


class FinanceGlobSpider(scrapy.Spider):
    name = "finance_glob"

    def start_requests(self):
        base_url = 'http://www.finances.gov.tn/fr/'
        pages = [
            'presentation',
            'conseil-national-de-la-fiscalite-cnf',
            'la-ministre-des-finances',
            'conseil-national-de-la-comptabilite-cnc',
            'conseil-national-des-assurances-cna',
            'le-conseil-national-des-normes-des-comptes-publics-cnncp',
        ]

        for page in pages:
            url = f'{base_url}{page}'
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta = {
                    'url': url
                }
            )


        

    def parse(self, response):
        data = {}
        data['url'] = response.meta['url']
        paragraphs = response.css('p::text')
        for p in paragraphs:
            data['content'] = p.get()
            yield data
