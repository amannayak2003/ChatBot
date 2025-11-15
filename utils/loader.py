from pypdf import PdfReader

class PDFLoader:
    def __init__(self, file_paths: list):
        self.file_paths = file_paths

    def load_pdfs(self):
        documents = []

        for path in self.file_paths:
            reader = PdfReader(path)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            documents.append({
                "filename": path,
                "text": text
            })

        return documents
