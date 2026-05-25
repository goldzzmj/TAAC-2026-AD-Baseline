"""Taiji training job submission script for Phase0.

Uses keyboard.insert_text() to edit run.sh (clipboard approach doesn't work).
"""

import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_DIR = r"<PROJECT_ROOT>/Phase0-验证修正与种子固定"
MODEL_TRAINING_DIR = os.path.join(PROJECT_DIR, "Model_Training")
JOB_NAME = "Phase0-验证修正-时间切分"
JOB_DESCRIPTION = "Phase0: v3 baseline + time-split validation + deterministic seed (infrastructure fix)"
ALLOWED_EXTENSIONS = {'.py', '.sh', '.ps1', '.json'}
CDP_URL = "http://localhost:9222"
EXCLUDE_FILES = {"submit_taiji.py", "run_compact.sh", "run_ascii.sh", "run.sh",
                 "inspect_cm.js", "inspect_cm2.js", "set_cm_content.js"}


async def get_file_rows(page):
    rows = await page.locator('tr.el-table__row').all()
    result = []
    for row in rows:
        try:
            name = (await row.locator('td').first.inner_text()).strip()
            result.append((name, row))
        except Exception:
            pass
    return result


async def delete_all_deletable(page):
    deleted = []
    for _ in range(20):
        rows = await get_file_rows(page)
        found = False
        for name, row in rows:
            delete_btn = row.locator('button.el-button.is-link').nth(2)
            try:
                if await delete_btn.count() > 0 and await delete_btn.is_visible():
                    await delete_btn.click(timeout=3000)
                    await asyncio.sleep(0.5)
                    deleted.append(name)
                    found = True
                    break
            except Exception:
                pass
        if not found:
            break
    return deleted


async def upload_one_file(page, file_path):
    basename = os.path.basename(file_path)
    try:
        upload_btn = page.locator('button:has-text("Upload from Local")').first
        async with page.expect_file_chooser(timeout=10000) as fc_info:
            await upload_btn.click(timeout=5000)
        file_chooser = await fc_info.value
        await file_chooser.set_files(file_path)
        return True
    except Exception as e:
        err = str(e).encode('ascii', errors='replace').decode('ascii')[:100]
        print(f"  Upload error for {basename}: {err}")
    return False


async def wait_for_file(page, filename, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = await get_file_rows(page)
        if any(n == filename for n, _ in rows):
            return True
        await asyncio.sleep(1)
    return False


async def edit_run_sh(page, content):
    """Edit run.sh using keyboard.insert_text() - the only method that works."""
    rows = await get_file_rows(page)
    for name, row in rows:
        if name == 'run.sh':
            edit_btn = row.locator('button.el-button.is-link').nth(1)
            if await edit_btn.count() > 0 and await edit_btn.is_visible():
                await edit_btn.click(timeout=3000)
                print("  Opened run.sh editor")
                break
    else:
        print("  run.sh not found")
        return False

    await asyncio.sleep(2)

    cm = page.locator('.cm-content').first
    try:
        await cm.wait_for(state='visible', timeout=5000)
    except Exception:
        print("  No CodeMirror content found")
        return False

    # Click, select all, then insert new content
    await cm.click(timeout=5000)
    await asyncio.sleep(0.5)
    await page.keyboard.press('Control+a')
    await asyncio.sleep(0.3)

    try:
        await page.keyboard.insert_text(content)
        print("  insert_text() succeeded")
    except Exception as e:
        print(f"  insert_text() failed: {str(e)[:80]}, using type() fallback")
        await page.keyboard.press('Backspace')
        await asyncio.sleep(0.2)
        await page.keyboard.type(content, delay=0)
        print("  type() fallback done")

    await asyncio.sleep(1)

    # Verify content
    verify = await page.evaluate('''() => {
        const cm = document.querySelector('.cm-content');
        return cm ? cm.textContent : '';
    }''')
    if 'time_split' in verify:
        print(f"  Content verified OK ({len(verify)} chars, time_split found)")
    else:
        print(f"  WARNING: Content mismatch ({len(verify)} chars)")
        print(f"  First 200 chars: {verify[:200]}")

    # Save via Submit button in dialog
    dialog = page.locator('.el-dialog').first
    try:
        submit_btn = dialog.locator('button:has-text("Submit")').first
        if await submit_btn.count() > 0 and await submit_btn.is_visible():
            await submit_btn.click(timeout=3000)
            print("  Saved run.sh")
            await asyncio.sleep(1.5)
            return True
    except Exception as e:
        print(f"  Submit error: {str(e)[:80]}")

    for sel in ['.el-dialog__footer button.el-button--primary',
                '.el-dialog button.el-button--primary']:
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click(timeout=3000)
                print(f"  Saved via: {sel}")
                await asyncio.sleep(1.5)
                return True
        except Exception:
            pass

    print("  WARNING: No Submit button found")
    await page.keyboard.press('Escape')
    await asyncio.sleep(1)
    return False


async def main():
    from playwright.async_api import async_playwright

    upload_files = []
    for f in sorted(os.listdir(MODEL_TRAINING_DIR)):
        if f in EXCLUDE_FILES:
            continue
        ext = os.path.splitext(f)[1]
        if ext not in ALLOWED_EXTENSIONS:
            continue
        full_path = os.path.join(MODEL_TRAINING_DIR, f)
        upload_files.append(full_path)

    upload_basenames = [os.path.basename(f) for f in upload_files]
    print(f"[1] Upload files ({len(upload_files)}): {upload_basenames}")

    # Read ASCII-only run.sh content
    run_sh_file = os.path.join(MODEL_TRAINING_DIR, "run_ascii.sh")
    with open(run_sh_file, 'r', encoding='ascii') as f:
        run_sh_content = f.read()
    print(f"    run.sh: {len(run_sh_content)} chars, {len(run_sh_content.splitlines())} lines")

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(CDP_URL)
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else await context.new_page()
        print(f"[2] Connected. URL: {page.url}")

        print("[3] Navigating to create page...")
        await page.goto("https://taiji.algo.qq.com/training/create",
                         wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        try:
            ok = page.locator('button:has-text("OK")').first
            if await ok.count() > 0 and await ok.is_visible():
                await ok.click(timeout=3000)
                await asyncio.sleep(0.5)
        except Exception:
            pass

        print(f"[4] URL: {page.url}")

        print("[5] Filling job name...")
        for sel in ['input[placeholder*="Please enter"]', 'input[placeholder*="Name"]',
                    'input[type="text"]']:
            try:
                inp = page.locator(sel).first
                if await inp.count() > 0 and await inp.is_visible():
                    await inp.fill(JOB_NAME, timeout=5000)
                    print(f"  OK via: {sel}")
                    break
            except Exception:
                continue
        await asyncio.sleep(0.5)

        print("[6] Filling description...")
        for sel in ['textarea[placeholder*="Please enter"]', 'textarea']:
            try:
                inp = page.locator(sel).first
                if await inp.count() > 0 and await inp.is_visible():
                    await inp.fill(JOB_DESCRIPTION, timeout=5000)
                    print(f"  OK via: {sel}")
                    break
            except Exception:
                continue
        await asyncio.sleep(1)

        print("[7] Deleting all existing files...")
        rows_before = await get_file_rows(page)
        print(f"  Before: {[n for n, _ in rows_before]}")
        deleted = await delete_all_deletable(page)
        print(f"  Deleted: {deleted}")
        await asyncio.sleep(1)

        print("[8] Uploading files...")
        uploaded = []
        for file_path in upload_files:
            basename = os.path.basename(file_path)
            ok = await upload_one_file(page, file_path)
            if ok:
                appeared = await wait_for_file(page, basename, timeout=20)
                print(f"  OK: {basename} ({'confirmed' if appeared else 'pending'})")
                uploaded.append(basename)
            else:
                print(f"  FAIL: {basename}")
            await asyncio.sleep(1)
        print(f"  Result: {len(uploaded)}/{len(upload_files)} uploaded")

        print("[9] Editing run.sh via insert_text()...")
        edit_ok = await edit_run_sh(page, run_sh_content)
        print(f"  Edit result: {edit_ok}")

        final_rows = await get_file_rows(page)
        final_names = [n for n, _ in final_rows]
        print(f"  Final files: {final_names}")

        await asyncio.sleep(2)

        # Hide overlays
        await page.evaluate('''() => {
            document.querySelectorAll('.el-overlay').forEach(el => {
                el.style.display = 'none';
            });
        }''')
        await asyncio.sleep(0.5)

        ss_pre = os.path.join(PROJECT_DIR, "debug_pre_submit_phase0.png")
        await page.screenshot(path=ss_pre, full_page=True)
        print(f"[10] Pre-submit screenshot saved")

        print("[11] Submitting...")
        submitted = False
        for sel in [
            'form button:has-text("Submit")',
            '.el-main button:has-text("Submit")',
            '#app button:has-text("Submit")',
            'button:has-text("Submit")',
        ]:
            try:
                btns = await page.locator(sel).all()
                for btn in btns:
                    try:
                        if await btn.is_visible():
                            parent = await btn.evaluate_handle(
                                'el => el.closest(".el-overlay, .el-dialog")')
                            parent_cls = await parent.evaluate(
                                'el => el ? el.className : null') if parent else None
                            if parent_cls is None:
                                await btn.click(timeout=5000)
                                submitted = True
                                print(f"  Clicked Submit via: {sel}")
                                await asyncio.sleep(3)
                                break
                    except Exception:
                        continue
                if submitted:
                    break
            except Exception:
                continue

        if not submitted:
            print("  Trying force click...")
            try:
                btns = await page.locator('button:has-text("Submit")').all()
                for btn in btns:
                    try:
                        if await btn.is_visible():
                            await btn.click(force=True, timeout=5000)
                            submitted = True
                            print("  Force-clicked Submit")
                            await asyncio.sleep(3)
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        if not submitted:
            print("  FAIL: No submit button found")
            return

        print("[12] Navigating to training list...")
        await asyncio.sleep(2)
        await page.goto("https://taiji.algo.qq.com/training",
                         wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        print("[13] Clicking Run...")
        run_clicked = False
        try:
            job_el = page.locator(f'td:has-text("{JOB_NAME}")').last
            if await job_el.count() > 0:
                print(f"  Found job: {JOB_NAME}")
                row = job_el.locator('xpath=ancestor::tr')
                run_btn = row.locator('button:has-text("Run")').first
                if await run_btn.count() > 0:
                    await run_btn.click(timeout=5000)
                    run_clicked = True
                    print("  Run clicked")
                    await asyncio.sleep(2)
                    for cs in ['button:has-text("OK")', 'button:has-text("Confirm")']:
                        try:
                            c = page.locator(cs).first
                            if await c.count() > 0 and await c.is_visible():
                                await c.click(timeout=3000)
                                print("  Confirmed")
                                break
                        except Exception:
                            pass
        except Exception as e:
            print(f"  Run error: {e}")

        await asyncio.sleep(3)
        final_ss = os.path.join(PROJECT_DIR, "taiji_submit_final_phase0.png")
        await page.screenshot(path=final_ss, full_page=True)

        print("\n" + "=" * 60)
        print("REPORT")
        print("=" * 60)
        print(f"Job:      {JOB_NAME}")
        print(f"Uploaded: {uploaded}")
        print(f"Final:    {final_names}")
        print(f"Submit:   {submitted}")
        print(f"Run:      {run_clicked}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
