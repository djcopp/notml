; Catch: move the paddle with keys 4 and 6 to catch the falling blocks.
; A catch adds a point; a miss resets the score. One game tick every two frames (30 a second at 60 Hz).
; The paddle moves 2 pixels a tick and the block falls a row every other tick, so it takes about two
; seconds to fall and the paddle can cross the whole screen in that time: every block can be caught.
;   VA, VB  paddle x, y        VC, VD  block x, y        VE  score        V9  ticks since the block fell
;   V0-V4   scratch (score drawing), V5-V8 scratch

        LD VA, 28
        LD VB, 30
        LD VE, 0
        LD I, paddle
        DRW VA, VB, 1
        CALL score

new:    RND VC, 0x3F            ; block starts at a random column
        LD VD, 0
        LD I, block
        DRW VC, VD, 2

tick:   LD V5, DT               ; wait for the delay timer
        SE V5, 0
        JP tick
        LD V5, 2
        LD DT, V5

        LD I, paddle            ; move the paddle: erase, move, redraw
        DRW VA, VB, 1
        LD V6, 4
        SKNP V6
        CALL left
        LD V6, 6
        SKNP V6
        CALL right
        LD I, paddle
        DRW VA, VB, 1

        ADD V9, 1               ; the block only falls every other tick
        SE V9, 2
        JP tick
        LD V9, 0

        LD I, block             ; move the block down a row
        DRW VC, VD, 2
        ADD VD, 1
        SE VD, 29
        JP fall

        LD V7, VC               ; at the paddle row: caught if 0 <= block x - paddle x <= 7
        SUB V7, VA
        SE VF, 1
        JP miss
        LD V8, 7
        SUB V8, V7
        SE VF, 1
        JP miss
        CALL score              ; erase the old score, add one, draw the new one
        ADD VE, 1
        CALL score
        JP new

miss:   CALL score
        LD VE, 0
        CALL score
        JP new

fall:   DRW VC, VD, 2
        JP tick

left:   SE VA, 0                ; the paddle starts at an even x and moves 2, so it stops exactly at 0 and 56
        ADD VA, 0xFE            ; minus two
        RET

right:  SE VA, 56
        ADD VA, 2
        RET

score:  LD I, digits            ; draw the last two digits of VE at the top left (XOR, so twice erases)
        LD B, VE
        LD V2, [I]
        LD V3, 1
        LD V4, 1
        LD F, V1
        DRW V3, V4, 5
        LD V3, 6
        LD F, V2
        DRW V3, V4, 5
        RET

paddle: DB 0xFF
block:  DB 0xC0, 0xC0
digits: DB 0, 0, 0
