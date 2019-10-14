# 把pdf转为pic
# 用到了 pymypdf
import fitz
import os
from PIL import Image
import argparse

def pdf2pic(pdf_path, outdir, mtx=3, dpi=300, fmt='tif'):
    print(pdf_path)
    if outdir != '':
        pdf_path = os.path.split(pdf_path)[1]
        pdf_name = os.path.join(outdir, os.path.splitext(pdf_path)[0])
    else:
        pdf_name = os.path.splitext(pdf_path)[0]

    pages = fitz.open(pdf_path)
    for page in pages:
        mat = fitz.Matrix(mtx, mtx)
        pix = page.getPixmap(alpha = False, matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # pix.writePNG("page-{}.png".format(page.number))
        pdf_name = pdf_name + "." + fmt
        img.save(pdf_name, dpi=[dpi,dpi], quality=90)
    print("-> {}   size: {:.2f}MB".format(pdf_name, os.path.getsize(pdf_name) / 1e6))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='pdf to pic')
    parser.add_argument('-p', '--pdfpath', dest='pdfpath', help='pdf file or dir', type=str, default='')
    parser.add_argument('-o', '--outdir', dest='outdir', help='', type=str, default='')
    parser.add_argument('-m', '--mtx', dest='mtx', help='size scale of image, default: 3', type=float, default=3)
    parser.add_argument('-d', '--dpi', dest='dpi', help='dpi, default: 300', type=int, default=300)
    parser.add_argument('-f', '--format', dest='fmt', help='format, default: "tif"', type=str, default='tif')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if os.path.isdir(args.pdfpath):
        pdffiles = os.listdir(args.pdfpath)
        print(pdffiles)
        for pdf in pdffiles:
            pdf = os.path.join(args.pdfpath, pdf)
            if os.path.isfile(pdf) and os.path.splitext(pdf)[1] == '.pdf':
                pdf2pic(pdf, args.outdir, args.mtx, args.dpi, args.fmt)
        print('Done!')
    elif os.path.isfile(args.pdfpath):
        if os.path.splitext(args.pdfpath)[1] == '.pdf':
            pdf2pic(args.pdfpath, args.outdir, args.mtx, args.dpi, args.fmt)
        print('Done!')
    else:
        print('error!')



if __name__ == "__main__":
    main()
    # print(1e6)