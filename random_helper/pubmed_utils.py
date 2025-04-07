from Bio import Entrez

# Always set your email so NCBI knows who is making the request.
# Replace this with your actual email address.
Entrez.email = "your_email@example.com"

def fetch_pubmed_data(pubmed_id: str) -> dict:
    """
    Fetches author list and abstract text for a given PubMed ID.

    Args:
        pubmed_id: The PubMed ID string.

    Returns:
        A dictionary containing 'authors' (pipe-separated string)
        and 'abstract' (string). Returns empty strings if data
        is not found or in case of an error.
    """
    authors_str = ""
    abstract = ""

    try:
        # Fetch the PubMed record in XML format
        handle = Entrez.efetch(db="pubmed", id=pubmed_id, retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        # Check if records were fetched and contain the expected structure
        if records and "PubmedArticle" in records and records["PubmedArticle"]:
            article = records["PubmedArticle"][0].get("MedlineCitation", {}).get("Article", {})

            # Extract the author list
            author_list = []
            if "AuthorList" in article:
                for author in article["AuthorList"]:
                    # Check if the author has separate first and last names
                    if "ForeName" in author and "LastName" in author:
                        full_name = f"{author['ForeName']} {author['LastName']}"
                    elif "CollectiveName" in author:
                        full_name = author["CollectiveName"]
                    else:
                        # Fallback if name parts are missing, though unlikely with standard entries
                        full_name = author.get("LastName", "Name Not Available") # Prioritize LastName if ForeName is missing
                    author_list.append(full_name)
            authors_str = "|".join(author_list)

            # Extract the abstract
            if "Abstract" in article and "AbstractText" in article["Abstract"]:
                abstract_parts = article["Abstract"]["AbstractText"]
                # Combine all parts if there are multiple paragraphs
                # Ensure parts are strings before joining
                abstract = "\n".join(str(part) for part in abstract_parts)

    except Exception as e:
        # Log the error or handle it as needed
        print(f"Error fetching data for PubMed ID {pubmed_id}: {e}")
        # Return empty strings in case of error to avoid breaking downstream processing
        return {"authors": "", "abstract": ""}

    return {"authors": authors_str, "abstract": abstract}

# Example usage (optional, can be removed or commented out)
if __name__ == '__main__':
    test_pubmed_id = "31452104"
    data = fetch_pubmed_data(test_pubmed_id)
    print(f"Data for PubMed ID {test_pubmed_id}:")
    print("Authors:", data["authors"])
    print("\nAbstract:", data["abstract"])

    test_pubmed_id_no_abstract = "1" # Example with potentially different structure or missing data
    data_no_abstract = fetch_pubmed_data(test_pubmed_id_no_abstract)
    print(f"\nData for PubMed ID {test_pubmed_id_no_abstract}:")
    print("Authors:", data_no_abstract["authors"])
    print("\nAbstract:", data_no_abstract["abstract"])
