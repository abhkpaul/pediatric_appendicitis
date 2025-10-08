import pandas as pd
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import yaml

# Try importing various PDF libraries with fallbacks
try:
    import fitz  # PyMuPDF - often the best for text extraction

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFDataExtractor:
    def __init__(self, config_path="configs/config.yaml"):
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            # Default configuration if file not found
            self.config = {
                'data_extraction': {
                    'us_pdf_directory': '/Users/abhk/Git/pediatric_appendicitis/data/raw/pdfs/ultrasound/',
                    'or_pdf_directory': '/Users/abhk/Git/pediatric_appendicitis/data/raw/pdfs/operative/',
                    'us_output_csv': '/Users/abhk/Git/pediatric_appendicitis/data/raw/us_reports.csv',
                    'or_output_csv': '/Users/abhk/Git/pediatric_appendicitis/data/raw/or_reports.csv',
                    'consolidated_output_csv': '/Users/abhk/Git/pediatric_appendicitis/data/raw/merged_data.csv'
                }
            }
            logger.warning(f"Config file {config_path} not found. Using default configuration.")

        self.us_reports_data = []
        self.or_reports_data = []

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF using multiple methods with fallbacks.
        Returns extracted text or None if all methods fail.
        """
        text = None

        # Method 1: Try PyMuPDF (fitz) first - often most reliable:cite[1]:cite[6]
        if PYMUPDF_AVAILABLE:
            try:
                with fitz.open(pdf_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text() + "\n"
                if text.strip():
                    logger.debug(f"Successfully extracted text using PyMuPDF from {os.path.basename(pdf_path)}")
                    return text.strip()
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")

        # Method 2: Try pypdf:cite[6]
        if PYPDF_AVAILABLE and text is None:
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                if text.strip():
                    logger.debug(f"Successfully extracted text using pypdf from {os.path.basename(pdf_path)}")
                    return text.strip()
            except Exception as e:
                logger.warning(f"pypdf failed for {pdf_path}: {e}")

        # Method 3: Try pdfplumber:cite[3]:cite[8]
        if PDFPLUMBER_AVAILABLE and text is None:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    logger.debug(f"Successfully extracted text using pdfplumber from {os.path.basename(pdf_path)}")
                    return text.strip()
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_path}: {e}")

        logger.error(f"All extraction methods failed for {pdf_path}")
        return None

    def parse_ultrasound_report(self, text, filename):
        """Parse ultrasound_test report text and extract structured data."""
        try:
            report_data = {
                'report_id': filename,
                'report_text': text,
                'extraction_timestamp': datetime.now().isoformat()
            }

            # Extract exam date with multiple patterns
            date_patterns = [
                r'Date of exam.*?([A-Za-z]+\s+\d{1,2},\s+\d{4})',
                r'Exam Date.*?([A-Za-z]+\s+\d{1,2},\s+\d{4})',
                r'Date.*?([A-Za-z]+\s+\d{1,2},\s+\d{4})'
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    report_data['exam_date'] = date_match.group(1)
                    break

            # Extract clinical information
            clinical_match = re.search(r'CLINICAL INFORMATION[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                       re.IGNORECASE | re.DOTALL)
            if clinical_match:
                report_data['clinical_information'] = clinical_match.group(1).strip()

            # Extract findings
            findings_match = re.search(r'FINDINGS[:\s]*(.*?)(?=IMPRESSION|$)', text, re.IGNORECASE | re.DOTALL)
            if findings_match:
                findings_text = findings_match.group(1).strip()
                report_data['findings'] = findings_text

                # Extract specific measurements from findings:cite[3]
                appendix_match = re.search(r'appendix.*?(\d+\.?\d*)\s*mm', findings_text, re.IGNORECASE)
                if appendix_match:
                    report_data['appendix_diameter_mm'] = float(appendix_match.group(1))

                wall_match = re.search(r'wall.*?thick.*?(\d+\.?\d*)\s*mm', findings_text, re.IGNORECASE)
                if wall_match:
                    report_data['wall_thickness_mm'] = float(wall_match.group(1))

                # Check for specific findings
                report_data['has_free_fluid'] = 'free fluid' in findings_text.lower()
                report_data['has_fat_stranding'] = any(
                    term in findings_text.lower() for term in ['fat stranding', 'hyperechoic fat'])
                report_data['has_lymph_nodes'] = 'lymph nodes' in findings_text.lower()
                report_data['has_appendicolith'] = 'appendicolith' in findings_text.lower()

            # Extract impression and determine severity
            impression_match = re.search(r'IMPRESSION[:\s]*(.*?)(?=$|\n\n)', text, re.IGNORECASE | re.DOTALL)
            if impression_match:
                impression_text = impression_match.group(1).strip()
                report_data['impression'] = impression_text

                # Determine severity from impression
                impression_lower = impression_text.lower()
                if 'normal' in impression_lower or 'unremarkable' in impression_lower:
                    report_data['severity_grade'] = 'Normal'
                elif 'simple' in impression_lower or 'uncomplicated' in impression_lower:
                    report_data['severity_grade'] = 'Simple/Uncomplicated'
                elif 'complicated' in impression_lower or 'gangrenous' in impression_lower:
                    report_data['severity_grade'] = 'Gangrenous'
                elif 'perforated' in impression_lower or 'ruptured' in impression_lower or 'abscess' in impression_lower:
                    report_data['severity_grade'] = 'Perforated'
                else:
                    report_data['severity_grade'] = 'Uncertain'

            return report_data

        except Exception as e:
            logger.error(f"Error parsing ultrasound_test report {filename}: {e}")
            return None

    def parse_operative_report(self, text, filename):
        """Parse operative report text and extract structured data."""
        try:
            report_data = {
                'report_id': filename,
                'report_text': text,
                'extraction_timestamp': datetime.now().isoformat()
            }

            # Extract date
            date_match = re.search(r'Date[:\s]*(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if date_match:
                report_data['operation_date'] = date_match.group(1)

            # Extract surgeon
            surgeon_match = re.search(r'Surgeon[:\s]*(.*?)(?=\n|$)', text, re.IGNORECASE)
            if surgeon_match:
                report_data['surgeon'] = surgeon_match.group(1).strip()

            # Extract pre-operative diagnosis
            preop_match = re.search(r'Pre-operative diagnosis[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                    re.IGNORECASE | re.DOTALL)
            if preop_match:
                report_data['preop_diagnosis'] = preop_match.group(1).strip()

            # Extract post-operative diagnosis and determine severity
            postop_match = re.search(r'Post-operative diagnosis[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                     re.IGNORECASE | re.DOTALL)
            if postop_match:
                postop_text = postop_match.group(1).strip()
                report_data['postop_diagnosis'] = postop_text

                # Determine severity from post-operative diagnosis
                postop_lower = postop_text.lower()
                if 'normal' in postop_lower or 'unremarkable' in postop_lower:
                    report_data['severity_grade'] = 'Normal'
                elif 'simple' in postop_lower or 'uncomplicated' in postop_lower:
                    report_data['severity_grade'] = 'Simple/Uncomplicated'
                elif 'complicated' in postop_lower or 'gangrenous' in postop_lower:
                    report_data['severity_grade'] = 'Gangrenous'
                elif 'perforated' in postop_lower or 'ruptured' in postop_lower or 'abscess' in postop_lower:
                    report_data['severity_grade'] = 'Perforated'
                else:
                    report_data['severity_grade'] = 'Uncertain'

            # Extract operative findings
            findings_match = re.search(r'operative findings[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                       re.IGNORECASE | re.DOTALL)
            if findings_match:
                findings_text = findings_match.group(1).strip()
                report_data['operative_findings'] = findings_text

                # Extract specific findings
                report_data['found_perforation'] = any(
                    term in findings_text.lower() for term in ['perforated', 'perforation', 'ruptured'])
                report_data['found_abscess'] = 'abscess' in findings_text.lower()
                report_data['found_gangrene'] = 'gangrenous' in findings_text.lower()
                report_data['found_contamination'] = any(
                    term in findings_text.lower() for term in ['contamination', 'purulent', 'fibrinous'])

            return report_data

        except Exception as e:
            logger.error(f"Error parsing operative report {filename}: {e}")
            return None

    def process_ultrasound_reports(self, pdf_directory):
        """Process all ultrasound_test PDF reports in a directory."""
        logger.info(f"Processing ultrasound_test reports from: {pdf_directory}")

        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []

        successful_extractions = 0

        for pdf_file in tqdm(pdf_files, desc="Extracting Ultrasound Reports"):
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                report_data = self.parse_ultrasound_report(text, pdf_file.stem)
                if report_data:
                    self.us_reports_data.append(report_data)
                    successful_extractions += 1
            else:
                logger.warning(f"Could not extract text from {pdf_file.name}")

        logger.info(f"Successfully processed {successful_extractions}/{len(pdf_files)} ultrasound_test reports")
        return self.us_reports_data

    def process_operative_reports(self, pdf_directory):
        """Process all operative PDF reports in a directory."""
        logger.info(f"Processing operative reports from: {pdf_directory}")

        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []

        successful_extractions = 0

        for pdf_file in tqdm(pdf_files, desc="Extracting Operative Reports"):
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                report_data = self.parse_operative_report(text, pdf_file.stem)
                if report_data:
                    self.or_reports_data.append(report_data)
                    successful_extractions += 1
            else:
                logger.warning(f"Could not extract text from {pdf_file.name}")

        logger.info(f"Successfully processed {successful_extractions}/{len(pdf_files)} operative reports")
        return self.or_reports_data

    def save_to_csv(self, data, output_path):
        """
        Save extracted data to CSV file.
        FIXED: Proper DataFrame empty check:cite[1]
        """
        # Proper way to check if DataFrame is empty
        if data is None or (hasattr(data, 'empty') and data.empty):
            logger.warning(f"No data to save to {output_path}")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convert to DataFrame if it's a list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path} with {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV {output_path}: {e}")
            return False

    def generate_consolidated_dataset(self):
        """Generate the final consolidated dataset by merging US and OR reports."""
        if not self.us_reports_data or not self.or_reports_data:
            logger.error("Need to process both ultrasound_test and operative reports first")
            return None

        # Create DataFrames
        us_df = pd.DataFrame(self.us_reports_data)
        or_df = pd.DataFrame(self.or_reports_data)

        # Add report type identifiers
        us_df['report_type'] = 'ultrasound_test'
        or_df['report_type'] = 'operative'

        # Create patient ID from report ID (extract numbers)
        us_df['patient_id'] = us_df['report_id'].str.extract(r'(\d+)', expand=False)
        or_df['patient_id'] = or_df['report_id'].str.extract(r'(\d+)', expand=False)

        # Merge datasets on patient_id
        merged_df = pd.merge(
            us_df,
            or_df,
            on='patient_id',
            how='inner',
            suffixes=('_us', '_or')
        )

        logger.info(f"Consolidated dataset created with {len(merged_df)} patient records")
        return merged_df

    def run_full_extraction(self):
        """Run the complete PDF extraction pipeline."""
        config = self.config['data_extraction']

        # Process ultrasound_test reports
        logger.info("Starting ultrasound_test reports processing...")
        us_data = self.process_ultrasound_reports(config['us_pdf_directory'])
        if us_data:
            success = self.save_to_csv(us_data, config['us_output_csv'])
            if success:
                logger.info(f"Ultrasound reports saved to {config['us_output_csv']}")

        # Process operative reports
        logger.info("Starting operative reports processing...")
        or_data = self.process_operative_reports(config['or_pdf_directory'])
        if or_data:
            success = self.save_to_csv(or_data, config['or_output_csv'])
            if success:
                logger.info(f"Operative reports saved to {config['or_output_csv']}")

        # Generate consolidated dataset
        if us_data and or_data:
            logger.info("Generating consolidated dataset...")
            consolidated_df = self.generate_consolidated_dataset()
            if consolidated_df is not None:
                success = self.save_to_csv(consolidated_df, config['consolidated_output_csv'])
                if success:
                    logger.info(f"Consolidated dataset saved to {config['consolidated_output_csv']}")
            else:
                logger.warning("Could not generate consolidated dataset - no matching patient records")
        else:
            logger.warning("Insufficient data to generate consolidated dataset")

        logger.info("PDF data extraction completed!")


def main():
    """Main function to run the PDF extraction."""
    extractor = PDFDataExtractor()
    extractor.run_full_extraction()


if __name__ == "__main__":
    main()