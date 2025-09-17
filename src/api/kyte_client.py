import requests
from typing import List, Dict, Any

# URL base da API de preços da Kyte
KYTE_PRICES_API_BASE_URL = "https://kyte-prices.azurewebsites.net/plans/"

def _fetch_prices_for_country(country_code: str) -> Dict[str, Any]:
    """Função auxiliar para buscar dados de preços para um único país."""
    url = f"{KYTE_PRICES_API_BASE_URL}{country_code.upper()}"
    try:
        response = requests.get(url, timeout=10)
        # Lança um erro para respostas HTTP ruins (4xx ou 5xx)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao buscar preços da Kyte para o país '{country_code}': {e}")
        return {}

def generate_pricing_documents_from_api() -> List[Dict[str, Any]]:
    """
    Busca os dados de preços da API da Kyte para todos os países conhecidos,
    tratando 'default' com contexto explícito, e gera documentos de conhecimento estruturados.
    """
    print("💰 Buscando e processando dados de preços da API da Kyte...")
    
    # Lista de países baseada na estrutura do arquivo pricesPlansByCountryCode.js [4-14]
    specific_country_codes = ["BR", "MX", "MY", "PH", "US", "CA", "GB"]
    all_codes_to_fetch = specific_country_codes + ["default"]
    
    # Cria uma string legível dos países com preços próprios para usar no contexto
    excluded_countries_str = ", ".join(specific_country_codes)

    # Informações de enriquecimento extraídas dos artigos da Intercom
    # sobre pagamentos e planos
    country_details = {
        "BR": {
            "url": "https://www.kyte.com.br/planos"
        },
        "MX": {
            "url": "https://www.appkyte.com/precios"
        },
        "default": {
            "url": "https://www.kyteapp.com/pricing"
        }
    }

    documents = []

    for country_code in all_codes_to_fetch:
        pricing_data = _fetch_prices_for_country(country_code)

        if not pricing_data:
            print(f" -> Nenhum dado de preço encontrado para {country_code.upper()}, pulando.")
            continue

        details = country_details.get(country_code.upper(), country_details["default"])

        is_default_case = country_code == "default"
        
        location_description = "Internacional (USD)" if is_default_case else country_code.upper()
        print(f" -> Processando preços para: {location_description}")

        for plan_name, prices in pricing_data.items():
            monthly_price = prices.get("monthly")
            yearly_price = prices.get("yearly")

            if not monthly_price or not yearly_price:
                continue

            title = f"Preço do plano {plan_name.upper()} do Kyte - {location_description}"
            
            if is_default_case:
                content = (
                    f"Para todos os países, exceto {excluded_countries_str}, os preços internacionais são em dólar americano (USD). "
                    f"O plano {plan_name.upper()} custa {monthly_price} por mês na modalidade mensal ou {yearly_price} por ano na modalidade anual. "
                    f"Mais detalhes em: {details['url']}"
                )
            else:
                content = (
                    f"O plano {plan_name.upper()} para a região {location_description} custa {monthly_price} por mês "
                    f"na modalidade mensal ou {yearly_price} por ano na modalidade anual. Mais detalhes em: {details['url']}"
                )


            document = {
                "title": title,
                "content": content,
                "category": "billing_plans_and_pricing",
                "language": "pt-BR",
                "meta_data": {
                    "source_type": "kyte_pricing_api",
                    "article_id": f"pricing_{country_code}_{plan_name}",
                    "is_chunked": False,
                    "chunk_index": 0,
                    "plan": plan_name.upper()
                }
            }

            document = {
                "title":title,
                "content": content,
                "category": "billing_plans_and_pricing",
                "tags": [],
                "platform": [],
                "plans": plan_name.upper(),
                "country": "INTERNATIONAL" if is_default_case else country_code.upper(),
                "language": "pt-BR",
                "meta_data": {
                    "source_type": "kyte_pricing_api",
                    "source_file": "kyte_pricing_api",
                    "article_id": f"pricing_{country_code}_{plan_name}",
                    "article_index": [],
                    "is_chunked": False,
                    "chunk_index": 0
                }
            }
            documents.append(document)

    print(f"✅ Geração de documentos de preços concluída. Total de {len(documents)} documentos criados.")
    return documents