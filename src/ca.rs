extern crate rand;
extern crate rayon;
use self::rayon::prelude::*;

pub struct CellA {
    pub cells: Vec<bool>,
    width: usize,
    height: usize,
    length: usize,
}

impl CellA {
    pub fn new ( width: usize, height: usize, length: usize ) -> CellA {
        let cells = (0 .. width * height * length)
            .into_par_iter()
            .map(|_| {
                // to randomize
                // if x < width * height {
                    rand::random()
                // } else {
                //     false
                // }

                // to fill center
                // if x == (width * height * length / 2) {
                //     true
                // } else {
                //     false
                // }
            })
            .collect();

        CellA {
            cells,
            width,
            height,
            length
        }
    }

    pub fn next_gen ( &mut self ) {
        let new_cells = (0 .. self.width * self.height * self.length)
            .into_par_iter()
            .map(|idx| {
                if (idx >= self.width * self.height + self.width + 1) && (idx < (self.width * self.height * self.length) - (self.width * self.height) - self.width - 1) {
                    let alive = self.cells[idx];
                    let mut neighbors = 0;

                    if self.cells[idx + (self.width * self.height) + self.width + 1] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height) + self.width    ] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height) + self.width - 1] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height)              + 1] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height)                 ] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height)              - 1] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height) - self.width + 1] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height) - self.width    ] { neighbors += 1; }
                    if self.cells[idx + (self.width * self.height) - self.width - 1] { neighbors += 1; }
                    if self.cells[idx                              + self.width + 1] { neighbors += 1; }
                    if self.cells[idx                              + self.width    ] { neighbors += 1; }
                    if self.cells[idx                              + self.width - 1] { neighbors += 1; }
                    if self.cells[idx                                           + 1] { neighbors += 1; }
                    if self.cells[idx                                           - 1] { neighbors += 1; }
                    if self.cells[idx                              - self.width + 1] { neighbors += 1; }
                    if self.cells[idx                              - self.width    ] { neighbors += 1; }
                    if self.cells[idx                              - self.width - 1] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height) + self.width + 1] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height) + self.width    ] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height) + self.width - 1] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height)              + 1] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height)                 ] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height)              - 1] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height) - self.width + 1] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height) - self.width    ] { neighbors += 1; }
                    if self.cells[idx - (self.width * self.height) - self.width - 1] { neighbors += 1; }

                    if alive {
                        if neighbors >= 12 && neighbors <= 26 {
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        if neighbors >= 16 && neighbors <= 26 {
                            return true;
                        } else {
                            return false;
                        }
                    }
                } else {
                    return false
                }
            })
            .collect();

        self.cells = new_cells;
    }
}
