use std::path::{Path, PathBuf};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use flate2::bufread::GzDecoder;
use std::str;
use rayon::prelude::*;
use glob::glob;
use std::sync::RwLock;

#[macro_use]
extern crate log;
extern crate simple_logger;

#[macro_use]
extern crate clap;

struct csvEntry {
    header: Vec<String>,
    line: Vec<String>
}

impl csvEntry {
    fn col(&self, pat: &str) -> Option<String> {
        let col_idx = self.header.iter().position(|r| r == pat).unwrap();
        if col_idx > self.line.len() {
            return None
        }
        Some(self.line[col_idx].clone())
    }
}


fn read_csv(path: PathBuf, skip_line: usize) -> impl std::iter::Iterator<Item = csvEntry> {
    let mut line_buf = String::with_capacity(100000);
    let mut csv_header: Vec<String> = Vec::new();
    let mut count: usize = 0;

    let fh = match fs::File::open(&path) {
        Err(e) => panic!("Couldn't open {}: {}", path.display(), e),
        Ok(f) => BufReader::with_capacity(10000 * 1024, f),
    };
    let mut fh: Box<dyn BufRead> = if path.extension().unwrap_or_default() == "gz" {
        Box::new(BufReader::with_capacity(
            1024 * 1024,
            GzDecoder::new(fh),
        ))
    } else {
        Box::new(fh)
    };
    std::iter::from_fn(move || {
        let result;
        line_buf.clear();

        if count == 0 {
            for i in 0..skip_line {
                fh.read_line(&mut line_buf).unwrap(); // Read the line
            }
            line_buf.clear();
            fh.read_line(&mut line_buf).unwrap(); // Read the line

            csv_header = line_buf.trim_end()
                .split(",").into_iter().map(|s| s.to_string()).collect();
            line_buf.clear();
        }

        fh.read_line(&mut line_buf).unwrap(); // Read the line
        let entry = csvEntry {
            header: csv_header.clone(),
            line: line_buf.trim_end()
            .split(",").into_iter().map(|s| s.to_string()).collect()
        };
        if line_buf.len() != 0 {
            result = Some(entry);
            count += 1;
        }
        else {
            result = None;
        }

        result
    })
}


fn main() {

    let root_folder = "oas_unpaired_210801/";
    let files_glob = glob(format!("{}*/*Heavy*.csv.gz", root_folder).as_str()).unwrap();
    let files: Vec<String> = files_glob.map(|f| f.unwrap().display().to_string() ).collect();

    let oas_out_file = "oas_processed_heavy";

    let mut oas_out_fh = BufWriter::with_capacity(1024*1024, fs::File::create(oas_out_file).unwrap());

    let mut oas_out_fh = RwLock::from(oas_out_fh);

    files.par_iter().for_each(|file| {
        println!("Processing: {}", file);

        let mut fileoutbuf = String::new();
        let mut seq_counter: usize = 0;
        
        for line in read_csv(PathBuf::from(file), 1) {
            let lc = line.col("sequence_alignment_aa");
            match lc {
                Some(s) => {
                    if !s.contains("X") {
                        fileoutbuf += format!("{}\n", &s).as_str();
                        seq_counter += 1;
                    }
                },
                None => {}
            }
        }
        if seq_counter > 0 {
            let mut w = oas_out_fh.write().unwrap();
            match w.write_all(fileoutbuf.as_bytes()) {
                Ok(..) => {},
                Err(e) => {
                    println!("{:?}", e);
                    panic!();
                }
            }
        }
        println!("Finished writing {} seqeunces to output file.", seq_counter);

    })

}
