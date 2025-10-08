import pandas as pd
import numpy as np
import os
import re
import PyPDF2
import pdfplumber
from pathlib import Path
from tqdm import tqdm
import logging
import yaml
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFDataExtractor:
    def __init__(self, config_path="/Users/abhk/Git/pediatric_appendicitis/configs/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.us_reports_data = []
        self.or_reports_data = []

    def extract_text_from_pdf(self, pdf_path, method='pdfplumber'):
        """
        Extract text from PDF using multiple methods for robustness.
        """
        text = ""
        try:
            if method == 'pdfplumber':
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
        except Exception as e:
            logger.warning(f"Error extracting text from {pdf_path}: {e}")
            return None

        return text.strip() if text else None

    def parse_ultrasound_report(self, text, filename):
        """
        Parse ultrasound_test report text and extract structured data.
        """
        try:
            report_data = {
                'report_id': filename,
                'report_text': text,
                'extraction_timestamp': datetime.now().isoformat()
            }

            # Extract exam date
            date_match = re.search(r'Date of exam.*?([A-Za-z]+\s+\d{1,2},\s+\d{4})', text, re.IGNORECASE)
            if date_match:
                report_data['exam_date'] = date_match.group(1)

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

                # Extract specific measurements from findings
                # Appendix diameter
                appendix_match = re.search(r'appendix.*?(\d+\.?\d*)\s*mm', findings_text, re.IGNORECASE)
                if appendix_match:
                    report_data['appendix_diameter_mm'] = float(appendix_match.group(1))

                # Wall thickness
                wall_match = re.search(r'wall.*?thick.*?(\d+\.?\d*)\s*mm', findings_text, re.IGNORECASE)
                if wall_match:
                    report_data['wall_thickness_mm'] = float(wall_match.group(1))

                # Check for specific findings
                report_data['has_free_fluid'] = 'free fluid' in findings_text.lower()
                report_data['has_fat_stranding'] = any(
                    term in findings_text.lower() for term in ['fat stranding', 'hyperechoic fat'])
                report_data['has_lymph_nodes'] = 'lymph nodes' in findings_text.lower()
                report_data['has_appendicolith'] = 'appendicolith' in findings_text.lower()

            # Extract impression
            impression_match = re.search(r'IMPRESSION[:\s]*(.*?)(?=$)', text, re.IGNORECASE | re.DOTALL)
            if impression_match:
                impression_text = impression_match.group(1).strip()
                report_data['impression'] = impression_text

                # Determine severity from impression
                impression_lower = impression_text.lower()
                if 'normal' in impression_lower or 'unremarkable' in impression_lower:
                    report_data['severity_grade'] = 'Normal'
                elif 'simple' in impression_lower or 'uncomplicated' in impression_lower or 'acute appendicitis' in impression_lower:
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
        """
        Parse operative report text and extract structured data.
        """
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

            # Extract post-operative diagnosis
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

            # Extract operation type
            operation_match = re.search(r'Operation[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                        re.IGNORECASE | re.DOTALL)
            if operation_match:
                report_data['operation_type'] = operation_match.group(1).strip()

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

            # Extract procedure details
            procedure_match = re.search(r'Operative procedure[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                        re.IGNORECASE | re.DOTALL)
            if procedure_match:
                report_data['procedure_details'] = procedure_match.group(1).strip()

            return report_data

        except Exception as e:
            logger.error(f"Error parsing operative report {filename}: {e}")
            return None

    def process_ultrasound_reports(self, pdf_directory):
        """
        Process all ultrasound_test PDF reports in a directory.
        """
        logger.info(f"Processing ultrasound_test reports from: {pdf_directory}")

        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return

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
        """
        Process all operative PDF reports in a directory.
        """
        logger.info(f"Processing operative reports from: {pdf_directory}")

        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return

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
        """
        if not data:
            logger.warning("No data to save")
            return False

        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path} with {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            return False

    def generate_consolidated_dataset(self):
        """
        Generate the final consolidated dataset by merging US and OR reports.
        """
        if not self.us_reports_data or not self.or_reports_data:
            logger.error("Need to process both ultrasound_test and operative reports first")
            return None

        # Create DataFrames
        us_df = pd.DataFrame(self.us_reports_data)
        or_df = pd.DataFrame(self.or_reports_data)

        # Add report type identifiers
        us_df['report_type'] = 'ultrasound_test'
        or_df['report_type'] = 'operative'

        # Create a common patient ID based on report ID patterns
        # This assumes report IDs follow a pattern that can be matched
        # Adjust this logic based on your actual file naming convention

        # For demonstration, we'll create a simple merge key
        us_df['patient_id'] = us_df['report_id'].str.extract(r'(\d+)')
        or_df['patient_id'] = or_df['report_id'].str.extract(r'(\d+)')

        # Merge datasets
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
        """
        Run the complete PDF extraction pipeline.
        """
        config = self.config['data_extraction']

        # Process ultrasound_test reports
        us_data = self.process_ultrasound_reports(config['us_pdf_directory'])
        if us_data:
            self.save_to_csv(us_data, config['us_output_csv'])

        # Process operative reports
        or_data = self.process_operative_reports(config['or_pdf_directory'])
        if or_data:
            self.save_to_csv(or_data, config['or_output_csv'])

        # Generate consolidated dataset
        if us_data and or_data:
            consolidated_df = self.generate_consolidated_dataset()
            if consolidated_df is not None:
                self.save_to_csv(consolidated_df, config['consolidated_output_csv'])

        logger.info("PDF data extraction completed!")


# Example usage and testing
if __name__ == "__main__":
    extractor = PDFDataExtractor()

    # Test with sample files
    sample_us_text = """
    US ABDOMEN/PELVIS- RIGHT LOWER QUADRANT TENDERNESS  
    Date of exam: September 12, 2019 23:04  

    CLINICAL INFORMATION: Fever, elevated white blood cell count and right lower quadrant pain. Rule out appendicitis.  

    FINDINGS: The appendix is dilated to 12mm, non-compressible, with wall thickening measuring 2.9mm. 
    Surrounding hyperechoic fat stranding is noted. A small amount of free fluid is present in the right lower quadrant.

    IMPRESSION: Findings are consistent with acute appendicitis.
    """

    sample_or_text = """
    OPERATIVE REPORT

    Date: 2024-05-23
    Surgeon: Dr. Sophie Chen

    Pre-operative diagnosis: COMPLICATED APPENDICITIS.

    Post-operative diagnosis: PERFORATED APPENDICITIS WITH ABSCESS.

    Operation: LAPAROSCOPIC APPENDECTOMY.

    Operative findings: The appendix was perforated with marked inflammation. A small localized abscess was found and drained.
    """

    print("Testing ultrasound_test report parsing:")
    us_data = extractor.parse_ultrasound_report(sample_us_text, "test_us_001")
    print(json.dumps(us_data, indent=2))

    print("\nTesting operative report parsing:")
    or_data = extractor.parse_operative_report(sample_or_text, "test_or_001")
    print(json.dumps(or_data, indent=2))