"""
Convert HTML presentation to PDF with each slide on one page.
Uses Playwright to render the HTML and capture each slide as a PDF page.
"""

from playwright.sync_api import sync_playwright
import time
import os

def convert_presentation_to_pdf(html_file, output_pdf):
    """
    Convert the HTML presentation to PDF with each slide on one page.
    
    Args:
        html_file: Path to the HTML presentation file
        output_pdf: Path to save the output PDF
    """
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Get absolute path to HTML file
        html_path = os.path.abspath(html_file)
        file_url = f"file:///{html_path.replace(os.sep, '/')}"
        
        print(f"Loading presentation from: {file_url}")
        page.goto(file_url)
        
        # Wait for the page to fully load
        time.sleep(2)
        
        # Enable dark theme by adding 'dark' class to html element
        page.evaluate("document.documentElement.classList.add('dark')")
        
        # Set color scheme to dark for proper rendering
        page.emulate_media(color_scheme='dark')
        
        # Wait for dark mode styles to apply
        time.sleep(1)
        
        # Verify dark mode is enabled
        has_dark_class = page.evaluate("document.documentElement.classList.contains('dark')")
        print(f"Dark theme enabled: {has_dark_class}")
        
        # Get total number of slides
        total_slides = page.evaluate("() => presentationData.length")
        print(f"Found {total_slides} slides")
        
        # Create a list to store individual slide PDFs
        temp_pdfs = []
        
        # Navigate through each slide and save as PDF
        for slide_num in range(total_slides):
            print(f"Processing slide {slide_num + 1}/{total_slides}...")
            
            # Navigate to the specific slide
            page.evaluate(f"currentSlide = {slide_num}; updateUI();")
            
            # Wait a moment for the slide to render
            time.sleep(0.5)
            
            # Generate PDF for this slide
            temp_pdf = f"temp_slide_{slide_num}.pdf"
            page.pdf(
                path=temp_pdf,
                format='Letter',  # Or 'A4' if you prefer
                print_background=True,
                prefer_css_page_size=False,
                page_ranges='1',
                margin={
                    'top': '0',
                    'right': '0',
                    'bottom': '0',
                    'left': '0'
                }
            )
            temp_pdfs.append(temp_pdf)
        
        browser.close()
        
        # Merge all PDFs into one
        print(f"\nMerging {len(temp_pdfs)} slides into {output_pdf}...")
        merge_pdfs(temp_pdfs, output_pdf)
        
        # Clean up temporary files
        for temp_pdf in temp_pdfs:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)
        
        print(f"✓ PDF saved to: {output_pdf}")

def merge_pdfs(pdf_list, output_path):
    """
    Merge multiple PDF files into a single PDF.
    
    Args:
        pdf_list: List of PDF file paths to merge
        output_path: Path to save the merged PDF
    """
    try:
        from PyPDF2 import PdfMerger
        
        merger = PdfMerger()
        for pdf in pdf_list:
            merger.append(pdf)
        
        merger.write(output_path)
        merger.close()
    except ImportError:
        # Fallback to pypdf if PyPDF2 is not available
        from pypdf import PdfMerger
        
        merger = PdfMerger()
        for pdf in pdf_list:
            merger.append(pdf)
        
        merger.write(output_path)
        merger.close()

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output paths
    html_file = os.path.join(script_dir, "index.html")
    output_pdf = os.path.join(script_dir, "presentation.pdf")
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file not found at {html_file}")
        exit(1)
    
    convert_presentation_to_pdf(html_file, output_pdf)
