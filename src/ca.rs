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
    pub fn new ( width: usize, height: usize, length: usize ) -> Self {
        let cells = (0 .. width * height * length)
            .into_par_iter()
            .map(|x| {
                // to randomize
                if x < width * height {
                    rand::random()
                } else {
                    false
                }

                // to fill center
                // if x == (width * height * length / 2) {
                //     true
                // } else {
                //     false
                // }
            })
            .collect();

        Self {
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
                if (idx > self.width * self.height + self.width) && (idx < (self.width * self.height * self.length) - (self.width * self.height) - self.width - 1) {
                    let alive = self.cells[idx];
                    let neighbors = [
                        self.cells[idx + (self.width * self.height) + self.width + 1],
                        self.cells[idx + (self.width * self.height) + self.width    ],
                        self.cells[idx + (self.width * self.height) + self.width - 1],
                        self.cells[idx + (self.width * self.height)              + 1],
                        self.cells[idx + (self.width * self.height)                 ],
                        self.cells[idx + (self.width * self.height)              - 1],
                        self.cells[idx + (self.width * self.height) - self.width + 1],
                        self.cells[idx + (self.width * self.height) - self.width    ],
                        self.cells[idx + (self.width * self.height) - self.width - 1],
                        self.cells[idx                              + self.width + 1],
                        self.cells[idx                              + self.width    ],
                        self.cells[idx                              + self.width - 1],
                        self.cells[idx                                           + 1],
                        self.cells[idx                                           - 1],
                        self.cells[idx                              - self.width + 1],
                        self.cells[idx                              - self.width    ],
                        self.cells[idx                              - self.width - 1],
                        self.cells[idx - (self.width * self.height) + self.width + 1],
                        self.cells[idx - (self.width * self.height) + self.width    ],
                        self.cells[idx - (self.width * self.height) + self.width - 1],
                        self.cells[idx - (self.width * self.height)              + 1],
                        self.cells[idx - (self.width * self.height)                 ],
                        self.cells[idx - (self.width * self.height)              - 1],
                        self.cells[idx - (self.width * self.height) - self.width + 1],
                        self.cells[idx - (self.width * self.height) - self.width    ],
                        self.cells[idx - (self.width * self.height) - self.width - 1]
                    ].iter().filter(|x| **x).count();

                    if alive { neighbors >= 4 && neighbors <= 5 }
                    else { neighbors >= 5 && neighbors <= 6 }
                } else {
                    false
                }
            })
            .collect();

        self.cells = new_cells;
    }
}
